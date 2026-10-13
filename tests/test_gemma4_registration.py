# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from defuser.model_registry import MODEL_CONFIG
from defuser.modeling.model_patches import _MODEL_PATCH_REGISTRY, patch_gemma4_runtime


def test_gemma4_in_model_registry():
    # convert_model / replace_fused_blocks skip any model_type absent from
    # MODEL_CONFIG, so this entry is what enables gemma4 expert defusion.
    assert "gemma4" in MODEL_CONFIG


def test_gemma4_runtime_patch_registered():
    # Gemma4TextExperts uses the [E, 2*inter, hidden] layout the is_transposed
    # heuristic mis-detects; the runtime patch must be wired.
    assert _MODEL_PATCH_REGISTRY.get("gemma4") is patch_gemma4_runtime
