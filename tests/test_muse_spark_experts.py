# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import transformers
from torch import nn

from defuser import convert_model
from defuser.model_registry import MODEL_CONFIG
from defuser.modeling.moe_experts_interface import muse_spark_experts_forward
from defuser.utils.common import is_version_at_least

pytestmark = pytest.mark.skipif(
    not is_version_at_least(transformers.__version__, MODEL_CONFIG["muse_spark"]["min_transformers_version"]),
    reason="transformers too old for muse_spark",
)


@pytest.mark.parametrize("model_type", ["muse_spark", "muse_spark_text"])
@torch.no_grad()
def test_muse_spark_experts_match_eager(model_type):
    from transformers.models.muse_spark import modeling_muse_spark as muse_spark

    torch.manual_seed(0)
    config = muse_spark.MuseSparkTextConfig(
        hidden_size=16,
        num_attention_heads=2,
        moe_hidden_size=8,
        moe_intermediate_size=6,
        num_local_experts=5,
        num_experts_per_tok=3,
    )
    config._experts_implementation = "eager"
    native = muse_spark.MuseSparkMoE(config).to(torch.bfloat16).eval()
    for parameter in native.parameters():
        parameter.normal_(std=0.2)

    model = nn.Module()
    model.config = SimpleNamespace(model_type=model_type)
    model.mlp = deepcopy(native)
    assert convert_model(model)
    experts = model.mlp.experts
    assert experts.forward.__func__ is muse_spark_experts_forward
    assert not hasattr(experts, "gate_up_proj")

    states = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(model.mlp(states), native(states))

    # Index 5 (num_experts) is the expert-parallel sentinel.
    indices = torch.tensor([[3, 1, 5], [1, 0, 2]])
    weights = torch.rand(indices.shape)
    hidden = torch.randn(2, 8, dtype=torch.bfloat16)
    torch.testing.assert_close(experts(hidden, indices, weights), native.experts(hidden, indices, weights))
