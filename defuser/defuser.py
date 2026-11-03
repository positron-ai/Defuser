# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import importlib
from copy import deepcopy

from torch import nn

from defuser.model_registry import MODEL_CONFIG, PATCH
from defuser.modeling.model_patches import apply_model_class_patches, apply_model_patches
from defuser.modeling.update_module import update_module
from defuser.utils.common import (
    MIN_SUPPORTED_TRANSFORMERS_VERSION,
    is_version_at_least,
    warn_if_public_api_transformers_unsupported,
)
import transformers
from logbar import LogBar

logger = LogBar(__name__)

def get_checkpoint_conversion_mapping(model_type):
    """Return Defuser's checkpoint remapping rules for one registered model type."""
    from transformers import conversion_mapping

    if not hasattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping"):
        conversion_mapping.orig_get_checkpoint_conversion_mapping = conversion_mapping.get_checkpoint_conversion_mapping

    cfg = MODEL_CONFIG.get(model_type)
    if cfg and "checkpoint_mapping" in cfg:
        return deepcopy(cfg["checkpoint_mapping"])

    return conversion_mapping.orig_get_checkpoint_conversion_mapping(model_type)


class PatchError(Exception):
    """Raised when Defuser cannot patch a registered Transformers class."""

    pass


def _has_prebuilt_replacements(model: nn.Module, model_type: str) -> bool:
    """Detect models that were already instantiated with registry-backed replacements."""
    replacement_paths = MODEL_CONFIG[model_type].get(PATCH.REPLACE_MODULE, [])
    replacement_class_paths = {custom_path for _, custom_path in replacement_paths}
    if not replacement_class_paths:
        return False

    for module in model.modules():
        class_path = f"{module.__class__.__module__}.{module.__class__.__name__}"
        if class_path in replacement_class_paths:
            return True

    return False


def replace_fused_blocks(model_type: str) -> bool:
    """Patch supported HF model classes so future loads instantiate defused blocks."""
    if warn_if_public_api_transformers_unsupported("replace_fused_blocks()", logger):
        return False

    apply_model_class_patches(model_type)

    cfg = MODEL_CONFIG.get(model_type)
    if cfg is None:
        return False

    patched_any = False
    for orig_path, custom_path in cfg.get(PATCH.REPLACE_MODULE, []):
        orig_module_path, orig_class_name = orig_path.rsplit(".", 1)
        custom_module_path, custom_class_name = custom_path.rsplit(".", 1)

        try:
            orig_module = importlib.import_module(orig_module_path)
            custom_module = importlib.import_module(custom_module_path)
            # Validate class existence before patching
            if not hasattr(orig_module, orig_class_name):
                raise PatchError(f"Original class[{orig_class_name}] not found: {orig_module}")

            if not hasattr(custom_module, custom_class_name):
                raise PatchError(f"Custom class[{custom_class_name}] not found: {custom_module}")

            custom_class = getattr(custom_module, custom_class_name)
            setattr(orig_module, orig_class_name, custom_class)

            if is_version_at_least(transformers.__version__, MIN_SUPPORTED_TRANSFORMERS_VERSION):
                from transformers import conversion_mapping

                if not hasattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping"):
                    conversion_mapping.orig_get_checkpoint_conversion_mapping = (
                        conversion_mapping.get_checkpoint_conversion_mapping
                    )

                conversion_mapping.get_checkpoint_conversion_mapping = get_checkpoint_conversion_mapping
                transformers.modeling_utils.get_checkpoint_conversion_mapping = get_checkpoint_conversion_mapping
            logger.info(f"Patched {orig_path} -> {custom_path}")
            patched_any = True

        except Exception as e:
            if isinstance(e, PatchError):
                raise e

            logger.warning(f"Failed to patch {orig_path}: {e}")
            return False
    return patched_any


def check_model_compatibility(model: nn.Module) -> bool:
    """Validate model type and transformers version compatibility."""
    from defuser.compatibility import _model_compatibility_issue

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    issue = _model_compatibility_issue(model_type, transformers.__version__)
    if issue is not None and issue.code == "model_version":
        logger.warn(
            f"Skip conversion for model_type={model_type}: "
            f"{issue.message}, current version is {transformers.__version__}."
        )
    return issue is None


def convert_model(
        model: nn.Module,
        cleanup_original: bool = False,
        max_layers: int | None = None,
        filter: list[str] | None = None,
) -> bool:
    """Convert one loaded model in place from fused experts to defused modules."""
    if warn_if_public_api_transformers_unsupported("convert_model()", logger):
        return False

    if max_layers is not None and max_layers < 1:
        raise ValueError("max_layers must be >= 1 when provided")

    if not check_model_compatibility(model):
        return False

    from defuser.modeling.moe_experts_interface import plan_moe_quantization

    planned = plan_moe_quantization(model, filter_rules=filter, max_layers=max_layers)
    apply_model_patches(model, max_layers=max_layers, filter_rules=filter)

    already_prebuilt = _has_prebuilt_replacements(model, model.config.model_type)

    # Perform runtime defusing of fused projections
    # Split already-loaded fused modules (e.g., gate_up_proj/down_proj) into
    # independent expert layers: gate_proj / up_proj / down_proj
    update_module(
        model,
        cleanup_original=cleanup_original,
        max_layers=max_layers,
        filter_rules=filter,
    )

    return not already_prebuilt or bool(planned)

__all__ = ["convert_model", "replace_fused_blocks"]
