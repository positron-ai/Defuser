"""Structural diagnostics; numerical qualification still requires workload tests."""

from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class CompatibilityIssue:
    code: str
    message: str


@dataclass(frozen=True)
class CompatibilityReport:
    model_type: str | None
    versions: dict[str, str]
    generic_targets: tuple[str, ...] = ()
    issues: tuple[CompatibilityIssue, ...] = ()

    @property
    def compatible(self) -> bool:
        """Whether the implemented structural checks found a blocker."""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "compatible": self.compatible}


@dataclass(frozen=True)
class ConversionReport:
    status: Literal["converted", "already_defused", "not_applicable", "unsupported"]
    compatibility: CompatibilityReport
    converted_paths: tuple[str, ...] = ()
    already_defused_paths: tuple[str, ...] = ()
    issues: tuple[CompatibilityIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompatibilityError(ValueError):
    """A required conversion did not produce or find defused modules."""

    def __init__(self, report: ConversionReport) -> None:
        self.report = report
        reasons = report.issues or report.compatibility.issues
        super().__init__("; ".join(issue.message for issue in reasons) or report.status)


def _model_compatibility_issue(
    model_type: str | None, runtime_version: str
) -> CompatibilityIssue | None:
    from defuser.utils.common import (
        MIN_SUPPORTED_TRANSFORMERS_VERSION,
        is_version_at_least,
    )

    minimum = MIN_SUPPORTED_TRANSFORMERS_VERSION
    if not is_version_at_least(runtime_version, minimum):
        return CompatibilityIssue(
            "transformers_version", f"Requires transformers>={minimum}"
        )

    from defuser.model_registry import MODEL_CONFIG

    config = MODEL_CONFIG.get(model_type)
    if config is None:
        return CompatibilityIssue(
            "model_type", f"Unregistered model type: {model_type!r}"
        )
    minimum = config.get("min_transformers_version", minimum)
    if not is_version_at_least(runtime_version, minimum):
        return CompatibilityIssue(
            "model_version", f"{model_type} requires transformers>={minimum}"
        )
    return None


def check_compatibility(
    model: Any,
    *,
    max_layers: int | None = None,
    filter: list[str] | None = None,
) -> CompatibilityReport:
    """Check dependencies, registration, version floors and generic expert layout.

    Does not modify the model or register an experts backend. Custom runtime
    patchers are checked by conversion, not simulated by this preflight.
    Importing dependencies can initialize their normal module-level state.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    versions = {}
    issues = []
    for distribution, module in (
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("pypcre", "pcre"),
        ("logbar", "logbar"),
        ("packaging", "packaging"),
        ("tqdm", "tqdm"),
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
            if importlib.util.find_spec(module) is None:
                raise ModuleNotFoundError(module)
        except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError) as error:
            issues.append(
                CompatibilityIssue("missing_dependency", f"{distribution}: {error}")
            )
    if issues:
        return CompatibilityReport(model_type, versions, issues=tuple(issues))

    try:
        import transformers

        versions["transformers_runtime"] = transformers.__version__
        issue = _model_compatibility_issue(model_type, transformers.__version__)
        if issue is not None:
            return CompatibilityReport(model_type, versions, issues=(issue,))
        from defuser.modeling.moe_experts_interface import plan_moe_quantization

        if max_layers is not None and max_layers < 1:
            raise ValueError("max_layers must be >= 1 when provided")
        targets = plan_moe_quantization(
            model, filter_rules=filter, max_layers=max_layers
        )
        return CompatibilityReport(
            model_type, versions, tuple(name for name, _, _ in targets)
        )
    except (
        ImportError,
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        RuntimeError,
    ) as error:
        issues.append(
            CompatibilityIssue(
                "structural_contract", f"{type(error).__name__}: {error}"
            )
        )
    return CompatibilityReport(model_type, versions, issues=tuple(issues))


def _defused_paths(
    model: Any, max_layers: int | None, filter_rules: list[str] | None
) -> set[str]:
    from torch import nn

    from defuser.model_registry import MODEL_CONFIG, PATCH
    from defuser.modeling.replace_modules import ReplacementModuleBase
    from defuser.utils.common import (
        compile_module_name_filter,
        is_within_max_layers,
        matches_module_name_filter,
    )

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    replacements = {
        custom
        for _, custom in MODEL_CONFIG.get(model_type, {}).get(PATCH.REPLACE_MODULE, [])
    }
    selected = compile_module_name_filter(filter_rules)
    paths = set()
    for name, module in model.named_modules():
        if not is_within_max_layers(name, max_layers) or not matches_module_name_filter(
            name, selected
        ):
            continue
        class_path = f"{type(module).__module__}.{type(module).__name__}"
        marked = getattr(module, "_unfused_experts", False) or getattr(
            module, "_defuser_split_gate_up_runtime", False
        )
        replacement = class_path in replacements or isinstance(
            module, ReplacementModuleBase
        )
        if (marked or replacement) and any(
            isinstance(child, nn.Linear) for child in module.modules()
        ):
            paths.add(name)
    return paths


def convert_model_with_report(
    model: Any,
    cleanup_original: bool = False,
    max_layers: int | None = None,
    filter: list[str] | None = None,
    *,
    required: bool = False,
) -> ConversionReport:
    """Convert and report observed module changes rather than a nominal Boolean.

    ``required`` accepts conversion or existing defused modules, and raises for
    unsupported/zero-target requests. Runtime errors can leave partial changes;
    this API reports them and does not promise rollback or numerical parity.
    """
    compatibility = check_compatibility(model, max_layers=max_layers, filter=filter)
    if not compatibility.compatible:
        report = ConversionReport("unsupported", compatibility)
    else:
        from defuser.defuser import convert_model

        before = _defused_paths(model, max_layers, filter)
        issues = ()
        try:
            convert_model(model, cleanup_original, max_layers, filter)
        except (
            ImportError,
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
        ) as error:
            issues = (
                CompatibilityIssue(
                    "conversion_error", f"{type(error).__name__}: {error}"
                ),
            )
        after = _defused_paths(model, max_layers, filter)
        converted = tuple(sorted(after - before))
        status = (
            "unsupported"
            if issues
            else (
                "converted"
                if converted
                else "already_defused"
                if after
                else "not_applicable"
            )
        )
        report = ConversionReport(
            status, compatibility, converted, tuple(sorted(before & after)), issues
        )
    if required and report.status not in ("converted", "already_defused"):
        raise CompatibilityError(report)
    return report
