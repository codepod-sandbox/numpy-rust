"""Test configuration for NumPy compatibility tests.

Stubs out external dependencies that aren't available in RustPython.
"""
import sys
import types

# -- hypothesis (stub that marks tests as skipped) ---------------------------
import pytest as _pytest

_hyp_mod = types.ModuleType("hypothesis")


def _skip_given(*a, **kw):
    def decorator(fn):
        return _pytest.mark.skip(reason="hypothesis not available")(fn)
    return decorator


_hyp_mod.given = _skip_given
_hyp_mod.assume = lambda x: None
_hyp_mod.settings = lambda *a, **kw: lambda fn: fn

class _FakeStrategy:
    """Stub hypothesis strategy that supports | operator."""
    def __or__(self, other):
        return self
    def __ror__(self, other):
        return self
    def filter(self, *a, **kw):
        return self
    def map(self, *a, **kw):
        return self
    def flatmap(self, *a, **kw):
        return self

_fake = _FakeStrategy()

_hyp_strategies = types.ModuleType("hypothesis.strategies")
# Add strategy stubs so class-level decorators can reference them
_hyp_strategies.data = lambda: _fake
_hyp_strategies.integers = lambda **kw: _fake
_hyp_strategies.floats = lambda **kw: _fake
_hyp_strategies.text = lambda **kw: _fake
_hyp_strategies.sampled_from = lambda x: _fake
_hyp_strategies.one_of = lambda *a: _fake
_hyp_strategies.just = lambda x: _fake
_hyp_strategies.none = lambda: _fake
_hyp_strategies.lists = lambda *a, **kw: _fake
_hyp_strategies.tuples = lambda *a: _fake
_hyp_strategies.booleans = lambda: _fake
_hyp_strategies.composite = lambda fn: fn

_hyp_mod.strategies = _hyp_strategies
sys.modules["hypothesis"] = _hyp_mod
sys.modules["hypothesis.strategies"] = _hyp_strategies

_hyp_extra = types.ModuleType("hypothesis.extra")
_hyp_extra_np = types.ModuleType("hypothesis.extra.numpy")
# Add numpy hypothesis strategy stubs
_hyp_extra_np.arrays = lambda *a, **kw: _fake
_hyp_extra_np.integer_dtypes = lambda **kw: _fake
_hyp_extra_np.floating_dtypes = lambda **kw: _fake
_hyp_extra_np.array_shapes = lambda **kw: _fake
_hyp_extra_np.mutually_broadcastable_shapes = lambda **kw: _fake
_hyp_extra_np.from_dtype = lambda *a, **kw: _fake
sys.modules["hypothesis.extra"] = _hyp_extra
sys.modules["hypothesis.extra.numpy"] = _hyp_extra_np
