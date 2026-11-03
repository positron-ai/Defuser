# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import io
import os
import pickle
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import defuser
from defuser import convert_model_with_report
from defuser.modeling.moe_experts_interface import (
    _detect_expert_projections,
    _unfuse_experts_weights_inplace,
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


@pytest.mark.parametrize("family", ["qwen3", "llama4"])
@pytest.mark.parametrize("roundtrip", ["pickle", "torch", "deepcopy"])
@torch.no_grad()
def test_converted_experts_roundtrip(family: str, roundtrip: str) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if family == "qwen3":
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeConfig,
            Qwen3MoeExperts,
        )

        config = Qwen3MoeConfig(
            hidden_size=8, moe_intermediate_size=6, num_experts=4, num_experts_per_tok=2
        )
        config._experts_implementation = "eager"
        native = Qwen3MoeExperts(config).eval()
        args = (
            torch.randn(2, 8),
            torch.tensor([[0, 2], [1, 3]]),
            torch.tensor([[0.6, 0.4], [0.25, 0.75]]),
        )
    else:
        from transformers.models.llama4.modeling_llama4 import (
            Llama4TextConfig,
            Llama4TextExperts,
        )

        config = Llama4TextConfig(
            hidden_size=8,
            intermediate_size=6,
            num_local_experts=4,
            num_experts_per_tok=2,
        )
        native = Llama4TextExperts(config).eval()
        args = (torch.randn(8, 8),)
    for parameter in native.parameters():
        parameter.normal_(std=0.1)
    expected = native(*args)
    assert torch.count_nonzero(expected)
    model = nn.Module()
    model.config = config
    model.experts = deepcopy(native)
    assert convert_model_with_report(model, required=True).status == "converted"
    if roundtrip == "pickle":
        restored = pickle.loads(pickle.dumps(model))
    elif roundtrip == "torch":
        stream = io.BytesIO()
        torch.save(model, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
    else:
        restored = deepcopy(model)
    torch.testing.assert_close(restored.experts(*args), expected)
    for parameter in restored.experts.parameters():
        parameter.zero_()
    torch.testing.assert_close(model.experts(*args), expected)
    torch.testing.assert_close(restored.experts(*args), torch.zeros_like(expected))


@torch.no_grad()
def test_converted_experts_load_in_fresh_process(tmp_path: Path) -> None:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeConfig,
        Qwen3MoeExperts,
    )

    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        hidden_size=8, moe_intermediate_size=6, num_experts=4, num_experts_per_tok=2
    )
    config._experts_implementation = "eager"
    model = nn.Module()
    model.config = config
    model.experts = Qwen3MoeExperts(config).eval()
    for parameter in model.parameters():
        parameter.normal_(std=0.1)
    args = (
        torch.randn(2, 8),
        torch.tensor([[0, 2], [1, 3]]),
        torch.tensor([[0.6, 0.4], [0.25, 0.75]]),
    )
    expected = model.experts(*args)
    assert convert_model_with_report(model, required=True).status == "converted"
    payload = tmp_path / "converted.pt"
    torch.save({"model": model, "args": args, "expected": expected}, payload)
    script = """
import sys
import torch
torch.set_num_threads(1)
payload = torch.load(sys.argv[1], weights_only=False)
with torch.no_grad():
    actual = payload['model'].experts(*payload['args'])
torch.testing.assert_close(actual, payload['expected'])
"""
    subprocess.run(
        [sys.executable, "-B", "-c", script, str(payload)],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(defuser.__file__).resolve().parents[1]),
        },
    )


@pytest.mark.parametrize("hidden,intermediate", [(4, 7), (7, 4), (4, 4)])
@pytest.mark.parametrize("transposed", [False, True])
@torch.no_grad()
def test_split_expert_layouts_match_native(
    hidden: int, intermediate: int, transposed: bool
) -> None:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeConfig,
        Qwen3MoeExperts,
    )

    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        num_experts=4,
        num_experts_per_tok=2,
    )
    config._experts_implementation = "eager"
    native = Qwen3MoeExperts(config).eval()
    for parameter in native.parameters():
        parameter.normal_(std=0.1)
    model = nn.Module()
    model.config = config
    model.experts = deepcopy(native)
    experts = model.experts
    gate, up = experts.gate_up_proj.chunk(2, dim=1)
    del experts.gate_up_proj
    experts.gate_proj = nn.Parameter(gate.clone())
    experts.up_proj = nn.Parameter(up.clone())
    del experts.is_transposed
    if transposed:
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(
                experts,
                name,
                nn.Parameter(getattr(experts, name).transpose(1, 2).contiguous()),
            )
        experts.is_transposed = True
    states = torch.randn(2, hidden)
    indices = torch.tensor([[0, 2], [1, 3]])
    weights = torch.tensor([[0.6, 0.4], [0.25, 0.75]])
    expected = native(states, indices, weights)
    assert convert_model_with_report(model, required=True).status == "converted"
    torch.testing.assert_close(experts(states, indices, weights), expected)


@pytest.mark.parametrize("selection", ["selected", "filter", "max_layers"])
def test_missing_experts_interface_respects_selection(selection: str) -> None:
    script = """
import builtins
import sys
from types import ModuleType
import torch
from torch import nn
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextConfig, Llama4TextExperts,
)
torch.set_num_threads(1)
config = Llama4TextConfig(hidden_size=8, intermediate_size=6, num_local_experts=4)
model = nn.Module()
model.config = config
model.layers = nn.ModuleList([nn.Identity(), Llama4TextExperts(config)])
with torch.no_grad():
    for parameter in model.parameters():
        parameter.normal_(std=0.1)
before = {name: value.clone() for name, value in model.state_dict().items()}
original_import = builtins.__import__
def blocked(
    name: str, globals: dict | None = None, locals: dict | None = None,
    fromlist: tuple = (), level: int = 0,
) -> ModuleType:
    if name == 'transformers.integrations.moe' or (
        name == 'transformers.integrations' and 'moe' in fromlist
    ):
        raise ImportError('unavailable experts integration')
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = blocked
from defuser import convert_model, convert_model_with_report
selection = sys.argv[1]
scope = {}
if selection == 'filter':
    scope['filter'] = [r'^layers\\.0$']
elif selection == 'max_layers':
    scope['max_layers'] = 1
report = convert_model_with_report(model, **scope)
assert report.status == ('unsupported' if selection == 'selected' else 'not_applicable')
if selection == 'selected':
    assert report.compatibility.issues[0].code == 'structural_contract'
    try:
        convert_model(model)
    except RuntimeError:
        pass
    else:
        raise AssertionError('selected conversion requires the experts interface')
assert model.state_dict().keys() == before.keys()
for name, value in model.state_dict().items():
    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
"""
    subprocess.run(
        [sys.executable, "-B", "-c", script, selection],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(defuser.__file__).resolve().parents[1]),
        },
    )
