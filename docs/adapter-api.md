# Adapter API

Adapters are trusted local Python. They build one implementation and provide deterministic probes;
they cannot supply parameter mappings, packing orders, semantic labels, or transforms.

```python
class ModelAdapter(Protocol):
    adapter_id: str
    def build(self, *, device: torch.device | str, dtype: torch.dtype) -> nn.Module: ...
    def prepare(self, model: nn.Module) -> None: ...
    def example_inputs(self, *, seed: int, device: torch.device | str) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    def select_outputs(self, output: Any) -> Any: ...
    def dynamic_shapes(self) -> Any | None: ...
    def differentiable_inputs(self, *, seed: int, device: torch.device | str) -> tuple[tuple[Any, ...], dict[str, Any]] | None: ...
```

Load adapters with `module:attribute`. An attribute may be an adapter instance or class. `prepare`
should disable caches and stochastic behavior. Both adapters' probes must describe equivalent
inputs. v0.1 semantic-intermediate capture currently requires positional probe arguments.

Model adapters execute arbitrary user-provided Python code. Only run adapters you trust.
