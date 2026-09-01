# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch
import torch.nn as nn

from defuser.modeling.moe_experts_interface import (
    _detect_expert_projections,
    _unfuse_experts_weights_inplace,
    linear_loop_experts_forward,
)


class _UnknownExpertsWithLossProperty(nn.Module):
    """Regression fixture for expert detection on modules with unrelated properties."""

    def __init__(self) -> None:
        super().__init__()
        self.expert_weight = nn.Parameter(torch.randn(4, 8, 16))
        self.loss_property_accesses = 0

    @property
    def loss_function(self):
        self.loss_property_accesses += 1
        raise AssertionError("expert detection should not touch unrelated properties")


def test_detect_expert_projections_ignores_unrelated_properties() -> None:
    """Projection detection should ignore unrelated properties on expert fixtures."""
    module = _UnknownExpertsWithLossProperty()

    detected = _detect_expert_projections(module)

    assert detected == {
        "expert_weight": {"is_input_proj": True, "output_multiplier": 1},
    }
    assert module.loss_property_accesses == 0


class _BufferBackedExperts(nn.Module):
    """Exercise the buffer-backed fused expert path end to end."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("gate_up_proj", torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4))
        self.register_buffer("down_proj", torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3))


def test_unfuse_experts_supports_registered_buffers() -> None:
    """Buffer-backed fused experts should unfuse into per-expert Linear layers."""
    module = _BufferBackedExperts()
    expected_gate_proj = module.gate_up_proj[0, :3].clone()
    expected_up_proj = module.gate_up_proj[0, 3:].clone()
    expected_down_proj = module.down_proj[0].clone()

    changed = _unfuse_experts_weights_inplace(module, check_decorator=False)

    assert changed is True
    expert0 = getattr(module, "0")
    torch.testing.assert_close(expert0.gate_proj.weight, expected_gate_proj)
    torch.testing.assert_close(expert0.up_proj.weight, expected_up_proj)
    torch.testing.assert_close(expert0.down_proj.weight, expected_down_proj)


class _LoopExperts(nn.Module):
    """Minimal experts module matching linear_loop_experts_forward's contract."""

    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int, gated: bool = True) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = nn.SiLU()
        for expert_idx in range(num_experts):
            container = nn.Module()
            if gated:
                container.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            container.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            container.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
            self.add_module(str(expert_idx), container)


def _masked_reference_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """The pre-rewrite per-expert boolean-mask algorithm, kept as the bit-exactness oracle."""
    if hidden_states.dim() == 3:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        top_k_index = top_k_index.view(-1, top_k_index.size(-1))
        top_k_weights = top_k_weights.view(-1, top_k_weights.size(-1))
    else:
        batch_size, seq_len = None, None
        hidden_dim = hidden_states.size(-1)

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    sample_weights = top_k_weights.reshape(-1).to(hidden_states.dtype)
    expert_ids = top_k_index.reshape(-1)
    selected_hidden_states = hidden_states[token_idx]
    out_per_sample = torch.zeros(token_idx.size(0), hidden_dim, device=device, dtype=hidden_states.dtype)

    for expert_idx in range(module.num_experts):
        mask = expert_ids == expert_idx
        if not mask.any():
            continue
        expert_input = selected_hidden_states[mask]
        expert = getattr(module, str(expert_idx))
        if hasattr(expert, "gate_proj"):
            gate_out = expert.gate_proj(expert_input)
            up_out = expert.up_proj(expert_input)
        else:
            gate_out = None
            up_out = expert.up_proj(expert_input)
        gated_out = module.act_fn(gate_out) * up_out if gate_out is not None else module.act_fn(up_out)
        out_per_sample[mask] = expert.down_proj(gated_out).to(out_per_sample.dtype)

    out_per_sample = out_per_sample * sample_weights.unsqueeze(-1)
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    if batch_size is not None:
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
    return final_hidden_states


def _routing(num_tokens: int, num_experts: int, top_k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    top_k_index = torch.randint(0, num_experts, (num_tokens, top_k), generator=generator)
    top_k_weights = torch.rand(num_tokens, top_k, generator=generator)
    return top_k_index, top_k_weights


def test_sorted_gather_forward_is_bit_exact_with_masked_reference() -> None:
    """The sorted-gather rewrite must equal the masked loop bit-for-bit (torch.equal)."""
    torch.manual_seed(0)
    module = _LoopExperts(num_experts=8, hidden_dim=16, intermediate_dim=32)
    hidden_states = torch.randn(64, 16)
    top_k_index, top_k_weights = _routing(64, 8, 2, seed=1)
    # Duplicate expert ids inside one token's top_k exercise the reshape+sum accumulation.
    top_k_index[0, 1] = top_k_index[0, 0]

    got = linear_loop_experts_forward(module, hidden_states, top_k_index, top_k_weights)
    want = _masked_reference_forward(module, hidden_states, top_k_index, top_k_weights)

    assert torch.equal(got, want)


def test_sorted_gather_forward_is_bit_exact_for_3d_and_non_gated() -> None:
    torch.manual_seed(0)
    module = _LoopExperts(num_experts=5, hidden_dim=12, intermediate_dim=24, gated=False)
    hidden_states = torch.randn(3, 7, 12)
    top_k_index, top_k_weights = _routing(21, 5, 3, seed=2)
    top_k_index = top_k_index.view(3, 7, 3)
    top_k_weights = top_k_weights.view(3, 7, 3)

    got = linear_loop_experts_forward(module, hidden_states, top_k_index, top_k_weights)
    want = _masked_reference_forward(module, hidden_states, top_k_index, top_k_weights)

    assert got.shape == (3, 7, 12)
    assert torch.equal(got, want)


def test_sorted_gather_preserves_row_order_and_skips_empty_experts() -> None:
    """Capture hooks must see the same rows in the same order, and no call for empty experts.

    Hessian accumulation on quantized experts hangs off forward hooks, so the
    rewrite must not reorder an expert's rows (stable argsort) nor invoke an
    expert that received no tokens.
    """
    torch.manual_seed(0)
    num_experts, hidden_dim = 4, 8
    module = _LoopExperts(num_experts=num_experts, hidden_dim=hidden_dim, intermediate_dim=16)
    num_tokens, top_k = 32, 2
    generator = torch.Generator().manual_seed(3)
    # Route only to experts 0-2 so expert 3 stays empty.
    top_k_index = torch.randint(0, num_experts - 1, (num_tokens, top_k), generator=generator)
    top_k_weights = torch.rand(num_tokens, top_k, generator=generator)
    hidden_states = torch.randn(num_tokens, hidden_dim)

    seen_inputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_experts)}
    for expert_idx in range(num_experts):
        expert = getattr(module, str(expert_idx))
        expert.up_proj.register_forward_hook(
            lambda _mod, args, _out, idx=expert_idx: seen_inputs[idx].append(args[0].detach().clone())
        )

    linear_loop_experts_forward(module, hidden_states, top_k_index, top_k_weights)

    assert seen_inputs[num_experts - 1] == []
    expert_ids = top_k_index.reshape(-1)
    token_idx = torch.arange(num_tokens).unsqueeze(1).expand(-1, top_k).reshape(-1)
    for expert_idx in range(num_experts - 1):
        mask = expert_ids == expert_idx
        if not mask.any():
            assert seen_inputs[expert_idx] == []
            continue
        assert len(seen_inputs[expert_idx]) == 1
        expected_rows = hidden_states[token_idx[mask]]
        assert torch.equal(seen_inputs[expert_idx][0], expected_rows)
