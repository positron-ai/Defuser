"""CPU contracts for the local compatibility candidate; no model downloads."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace

import pytest
import torch
import transformers
from torch import nn

from defuser import (
    CompatibilityError,
    check_compatibility,
    convert_model,
    convert_model_with_report,
)
from defuser.modeling.moe_experts_interface import (
    _unfuse_experts_weights_inplace,
    validate_expert_layout,
)
from defuser.modeling.runtime_defusion import patch_dbrx_experts
from defuser.modeling.unfused_moe.common import resolve_topk_routing
from defuser.utils.common import is_version_at_least


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    monkeypatch.setattr(
        "defuser.modeling.moe_experts_interface.clear_memory", lambda: None
    )


def initialize(module: nn.Module) -> nn.Module:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.1)
        for buffer in module.buffers():
            if buffer.is_floating_point():
                buffer.zero_()
    return module.eval()


def qwen_model(saved_backend: str | None = None, layers: int = 1) -> nn.Module:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeConfig,
        Qwen3MoeExperts,
    )

    config = Qwen3MoeConfig(
        hidden_size=8,
        moe_intermediate_size=6,
        num_experts=4,
        num_experts_per_tok=2,
    )
    config._experts_implementation = "eager"
    if saved_backend is not None:
        config.experts_implementation = saved_backend
    model = nn.Module()
    model.config = config
    model.layers = nn.ModuleList(
        [initialize(Qwen3MoeExperts(config)) for _ in range(layers)]
    )
    return model


@pytest.mark.parametrize("family", ["glm4_moe", "glm4_moe_lite"])
@torch.no_grad()
def test_glm_router_same_weights_parity(family: str) -> None:
    module = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    replacement = importlib.import_module(f"defuser.modeling.unfused_moe.{family}")
    prefix = "Glm4MoeLite" if family.endswith("lite") else "Glm4Moe"
    config = getattr(module, f"{prefix}Config")(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        n_routed_experts=4,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
    )
    config._experts_implementation = "eager"
    native = initialize(getattr(module, f"{prefix}MoE")(config))
    defused = getattr(replacement, f"Linear{prefix}MoE")(config).eval()
    defused.gate.load_state_dict(native.gate.state_dict())
    defused.shared_experts.load_state_dict(native.shared_experts.state_dict())
    for index, expert in enumerate(defused.experts):
        gate, up = native.experts.gate_up_proj[index].chunk(2, dim=0)
        expert.gate_proj.weight.copy_(gate)
        expert.up_proj.weight.copy_(up)
        expert.down_proj.weight.copy_(native.experts.down_proj[index])
    states = torch.randn(2, 3, 8)
    torch.testing.assert_close(defused(states), native(states), rtol=1e-5, atol=1e-7)


def test_router_triple_uses_supplied_choices() -> None:
    indices = torch.tensor([[3, 1]])
    weights = torch.tensor([[0.2, 0.8]])

    def forbidden(_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("new router output must not be recomputed")

    actual_indices, actual_weights = resolve_topk_routing(
        (torch.zeros(1, 4), weights, indices), forbidden
    )
    assert actual_indices is indices
    assert actual_weights is weights
    with pytest.raises(TypeError, match="router output"):
        resolve_topk_routing((weights, indices), forbidden)


def test_saved_backend_rejected_without_mutation() -> None:
    model = qwen_model("eager")
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    diagnostics = check_compatibility(model)
    assert not diagnostics.compatible
    assert "Backend 'eager'" in diagnostics.issues[0].message
    with pytest.raises(ValueError, match="Backend 'eager'"):
        convert_model(model)
    assert before.keys() == model.state_dict().keys()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
    assert model.config._experts_implementation == "eager"


def test_default_gate_requires_callable_activation() -> None:
    model = qwen_model()
    model.layers[0].act_fn = None
    fused = model.layers[0].gate_up_proj
    report = convert_model_with_report(model)
    assert report.status == "unsupported"
    assert "act_fn" in report.compatibility.issues[0].message
    assert model.layers[0].gate_up_proj is fused


@torch.no_grad()
def test_noncallable_gate_uses_valid_activation() -> None:
    model = qwen_model()
    states = torch.randn(2, 8)
    indices = torch.tensor([[0, 2], [1, 3]])
    weights = torch.tensor([[0.6, 0.4], [0.25, 0.75]])
    expected = model.layers[0](states, indices, weights)
    model.layers[0]._apply_gate = None
    assert convert_model_with_report(model, required=True).status == "converted"
    torch.testing.assert_close(model.layers[0](states, indices, weights), expected)


def test_invalid_filter_is_structured() -> None:
    report = convert_model_with_report(qwen_model(), filter=["["])
    assert report.status == "unsupported"
    assert "Invalid module filter" in report.compatibility.issues[0].message


def test_prebuilt_block_does_not_skip_other_targets() -> None:
    from defuser.modeling.unfused_moe.qwen3_moe import LinearQwen3MoeSparseMoeBlock

    model = qwen_model(layers=2)
    model.layers[0] = LinearQwen3MoeSparseMoeBlock(model.config)
    assert check_compatibility(model).generic_targets == ("layers.1",)
    result = convert_model_with_report(model, filter=[r"^layers\.1$"], required=True)
    assert result.status == "converted"
    assert result.converted_paths == ("layers.1",)
    assert not hasattr(model.layers[1], "gate_up_proj")


@torch.no_grad()
def test_conversion_report_scope_and_parity() -> None:
    model = qwen_model(layers=2)
    native = deepcopy(model.layers[0])
    unselected = deepcopy(model.layers[1])
    states = torch.randn(2, 8)
    indices = torch.tensor([[0, 2], [1, 3]])
    weights = torch.tensor([[0.6, 0.4], [0.25, 0.75]])
    expected = native(states, indices, weights)
    before = model.layers[0].gate_up_proj
    diagnostics = check_compatibility(model, max_layers=1)
    assert diagnostics.generic_targets == ("layers.0",)
    assert model.layers[0].gate_up_proj is before
    report = convert_model_with_report(model, max_layers=1, required=True)
    assert report.status == "converted"
    assert report.converted_paths == ("layers.0",)
    assert hasattr(model.layers[1], "gate_up_proj")
    torch.testing.assert_close(
        model.layers[1](states, indices, weights), unselected(states, indices, weights)
    )
    torch.testing.assert_close(model.layers[0](states, indices, weights), expected)
    again = convert_model_with_report(model, max_layers=1, required=True)
    assert again.status == "already_defused"
    assert again.already_defused_paths == ("layers.0",)
    json.dumps(again.to_dict())


def test_report_zero_targets_and_unregistered() -> None:
    model = nn.Module()
    model.config = SimpleNamespace(model_type="gemma4")
    model.dense = nn.Linear(8, 8)
    assert convert_model_with_report(model).status == "not_applicable"
    with pytest.raises(CompatibilityError) as error:
        convert_model_with_report(model, required=True)
    assert error.value.report.status == "not_applicable"
    model.config.model_type = "unknown_fixture"
    assert convert_model_with_report(model).status == "unsupported"


@pytest.mark.parametrize("damage", ["odd_split", "count", "down", "bias", "unknown"])
def test_invalid_layout_leaves_all_weights_intact(damage: str) -> None:
    model = qwen_model(layers=2)
    target = model.layers[1]
    if damage == "odd_split":
        target.gate_up_proj = nn.Parameter(torch.randn(4, 11, 8))
    elif damage == "count":
        target.down_proj = nn.Parameter(torch.randn(3, 8, 6))
    elif damage == "down":
        target.down_proj = nn.Parameter(torch.randn(4, 9, 6))
    elif damage == "bias":
        target.register_parameter("gate_up_proj_bias", nn.Parameter(torch.randn(4, 11)))
    else:
        del target.gate_up_proj
        del target.down_proj
        target.register_parameter("mystery_weight", nn.Parameter(torch.randn(4, 6, 8)))
    before = dict(model.named_parameters())
    with pytest.raises(ValueError):
        convert_model(model)
    assert before.keys() == dict(model.named_parameters()).keys()
    assert all(
        parameter is dict(model.named_parameters())[name]
        for name, parameter in before.items()
    )


def test_standard_layout_when_intermediate_is_small() -> None:
    experts = nn.Module()
    experts.gate_up_proj = nn.Parameter(torch.randn(3, 6, 16))
    experts.down_proj = nn.Parameter(torch.randn(3, 16, 3))
    gate = experts.gate_up_proj[0, :3].clone()
    assert validate_expert_layout(experts) is False
    assert _unfuse_experts_weights_inplace(experts, check_decorator=False)
    torch.testing.assert_close(experts[0].gate_proj.weight, gate)


def test_unrelated_forward_wrapper_is_not_experts_dispatch() -> None:
    class Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(4, 12, 8))
            self.down_proj = nn.Parameter(torch.randn(4, 8, 6))

        @wraps(nn.Module.forward)
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value

    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_moe")
    model.experts = Wrapped()
    assert convert_model_with_report(model).status == "not_applicable"
    assert hasattr(model.experts, "gate_up_proj")


@torch.no_grad()
def test_dbrx_projection_reference() -> None:
    from transformers.models.dbrx.modeling_dbrx import DbrxExperts

    native = initialize(
        DbrxExperts(
            SimpleNamespace(
                hidden_size=8,
                ffn_hidden_size=16,
                moe_num_experts=4,
                ffn_act_fn={"name": "silu"},
            )
        )
    )
    changed = deepcopy(native)
    states = torch.randn(1, 2, 8)
    indices = torch.tensor([[0, 2], [1, 3]])
    weights = torch.tensor([[0.6, 0.4], [0.25, 0.75]])
    expected = torch.zeros_like(states).view(2, 8)
    w1 = native.mlp.w1.view(4, 16, 8)
    v1 = native.mlp.v1.view(4, 16, 8)
    w2 = native.mlp.w2.view(4, 16, 8)
    for token in range(2):
        for rank in range(2):
            expert = indices[token, rank]
            hidden = states.view(2, 8)[token]
            gate = torch.nn.functional.silu(hidden @ w1[expert].T)
            up = hidden @ v1[expert].T
            expected[token] += ((gate * up) @ w2[expert]) * weights[token, rank]
    assert patch_dbrx_experts(changed)
    torch.testing.assert_close(
        changed(states, indices, weights), expected.view_as(states)
    )
    malformed = deepcopy(native)
    malformed.mlp.w2 = nn.Parameter(torch.randn(4, 16, 9))
    with pytest.raises(ValueError, match="DBRX w2"):
        patch_dbrx_experts(malformed)
    assert hasattr(malformed, "mlp")
    assert "0" not in malformed._modules


@torch.no_grad()
def test_dbrx_same_weights_parity() -> None:
    if not is_version_at_least(transformers.__version__, "5.20.0"):
        pytest.skip("older native DBRX has its own unequal-width forward error")
    from transformers.models.dbrx.modeling_dbrx import DbrxExperts

    native = initialize(
        DbrxExperts(
            SimpleNamespace(
                hidden_size=8,
                ffn_hidden_size=16,
                moe_num_experts=4,
                ffn_act_fn={"name": "silu"},
            )
        )
    )
    defused = deepcopy(native)
    assert patch_dbrx_experts(defused)
    assert not patch_dbrx_experts(defused)
    states = torch.randn(1, 2, 8)
    indices = torch.tensor([[0, 2], [1, 3]])
    weights = torch.tensor([[0.6, 0.4], [0.25, 0.75]])
    torch.testing.assert_close(
        defused(states, indices, weights), native(states, indices, weights)
    )
    with pytest.raises(ValueError, match="final dimension"):
        defused(torch.randn(1, 2, 16), indices, weights)


def test_diagnostics_import_without_optional_runtime() -> None:
    script = """
import importlib.abc
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'pcre', 'logbar'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockRuntime())
import defuser
report = defuser.check_compatibility(object())
assert not report.compatible
assert all(issue.code == 'missing_dependency' for issue in report.issues)
"""
    subprocess.run(
        [sys.executable, "-B", "-c", script],
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
