# Compatibility checks

The `0.0.28+positron.3.dev0` candidate builds on `204a1f04`.
It is an experimental artifact, not a published release or a replacement for
workload qualification.

```python
from defuser import check_compatibility, convert_model_with_report

diagnostics = check_compatibility(model, max_layers=1)
print(diagnostics.to_dict())
result = convert_model_with_report(model, max_layers=1, required=True)
print(result.to_dict())
```

The diagnostic API checks installed dependencies, the model registry, version
floors, and selected generic experts' projection geometry, routing interface,
activation and explicit backend settings. It does not mutate model tensors or
register a backend. Importing dependencies can initialize their own global state.
Version eligibility uses the loaded Transformers runtime; reports retain its
version as `transformers_runtime` alongside the distribution metadata.
`generic_targets` covers the generic pass; it is not an exhaustive list of custom
runtime patches. Custom patchers validate their inputs when executed.

Conversion reports distinguish `converted`, `already_defused`, `not_applicable`
and `unsupported`, with observed module paths and diagnostic reasons. Paths use
the names from `named_modules()`; the empty string denotes the root module.
`required=True` rejects unsupported and zero-target calls. It also accepts
existing defused modules within the selected scope. It does not assert that
every requested layer was converted or numerically qualified. The legacy
`convert_model()` Boolean interface remains available, including its historical
no-op behavior. Callers requiring conversion should use the report API.

Generic expert targets are validated together before their tensors are split.
An incompatible saved public `experts_implementation`, including `eager`, is a
blocker. Make an explicit choice in a disposable model configuration before
retrying; Defuser does not silently reinterpret that saved setting. The private
`_experts_implementation='eager'` initialization setting remains supported.
Converted instances receive their own forward method so filtered conversion does
not switch the backend of untouched modules sharing the same configuration.
The generic expert forward bindings support pickle and whole-model serialization.
Consistent split
gate/up/down projections use standard `[expert, out, in]` weights when no
`is_transposed` flag is present; transposed weights declare that flag as `True`.

The GLM4 MoE and Lite replacements accept legacy logits-only routers and newer
`(logits, weights, indices)` routers. Newer routing weights and choices are used
directly. DBRX uses hidden-to-FFN gate/up projections and FFN-to-hidden down
projections, retaining the input shape. The tests compare native and defused
outputs using identical weights. Muse Spark keeps its dedicated forward and
BF16/sentinel regression test.

Runtime patching is not transactional: allocation errors or a custom patch error
can leave a partly modified model. Reports include observed changes on such an
error; discard that instance before retrying. These checks do not qualify GPU
kernels, quantized output, full-model KLD, checkpoint loading or every model
registered by Defuser.

Run the focused CPU contracts in an isolated environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_compatibility.py tests/test_moe_experts_interface.py \
  tests/test_muse_spark_experts.py
```

The package requests `transformers[torch]` so each selected Transformers release
enforces its own Torch dependency floor. Exact Torch/CUDA builds remain the
responsibility of environment locks.

Build a wheel with `python -m build --wheel`. The proposed CPU workflow exercises
the same contracts across four Transformers versions on Python 3.14, serially
and with eight pytest workers.
Workflow results and local test results have separate validation scopes.
