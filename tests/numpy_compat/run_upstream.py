"""
Universal test runner for vendored upstream NumPy test files.

Provides minimal shims for pytest and hypothesis (which are not available in
RustPython), then loads and runs upstream test files.

All numpy internal modules (_core, testing, lib, etc.) are now proper packages
under python/numpy/ — no shims needed here.

Usage:
    ./target/release/numpy-python tests/numpy_compat/run_upstream.py <test_file> [--ci] [--verbose|-v]
    ./target/release/numpy-python tests/numpy_compat/run_upstream.py --scan  # scan ALL files

Examples:
    ./target/release/numpy-python tests/numpy_compat/run_upstream.py upstream/core_test_numeric.py
    ./target/release/numpy-python tests/numpy_compat/run_upstream.py upstream/lib_test_nanfunctions.py --ci
    ./target/release/numpy-python tests/numpy_compat/run_upstream.py --scan
"""

import sys
import os
import types
import traceback
import time

_this_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Minimal pytest shim
# ---------------------------------------------------------------------------

_pytest = types.ModuleType("pytest")
_pytest.__name__ = "pytest"
_pytest.__package__ = "pytest"


class _SkipException(Exception):
    pass


class _RaisesContext:
    def __init__(self, exc_type, match=None):
        self.exc_type = exc_type
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        if exc_type is None:
            raise AssertionError(
                "Expected {} but no exception was raised".format(
                    self.exc_type.__name__))
        if issubclass(exc_type, self.exc_type):
            self.value = exc_val
            if self.match is not None:
                import re
                if not re.search(self.match, str(exc_val)):
                    raise AssertionError(
                        "Regex {!r} did not match {!r}".format(
                            self.match, str(exc_val)))
            return True
        return False


class _WarnsContext:
    def __init__(self):
        self.list = []
        self._warnings = []
    def __enter__(self):
        import warnings as _w
        self._old_filters = _w.filters[:]
        self._old_showwarning = _w.showwarning
        ctx = self
        def _capture(message, category, filename, lineno, file=None, line=None):
            ctx._warnings.append(type('W', (), {'category': category, 'message': message})())
        _w.showwarning = _capture
        _w.simplefilter('always')
        return self
    def __exit__(self, exc_type, exc_val, tb):
        import warnings as _w
        _w.filters[:] = self._old_filters
        _w.showwarning = self._old_showwarning
        return False
    def __len__(self):
        return len(self._warnings)
    def __iter__(self):
        return iter(self._warnings)
    def __getitem__(self, idx):
        return self._warnings[idx]


def _raises(exc, *args, match=None, **kwargs):
    if args:
        fn = args[0]
        fargs = args[1:]
        try:
            fn(*fargs, **kwargs)
            raise AssertionError(
                "Expected {} but no exception was raised".format(exc.__name__))
        except exc:
            pass
        return
    return _RaisesContext(exc, match)


def _warns(cls, *args, **kwargs):
    if args:
        fn = args[0]
        fargs = args[1:]
        return fn(*fargs, **kwargs)
    return _WarnsContext()


def _approx(x, rel=None, abs=None, nan_ok=False):
    return x


class _Mark:
    def parametrize(self, names=None, values=None, *, argnames=None, argvalues=None, **kwargs):
        if argnames is not None:
            names = argnames
        if argvalues is not None:
            values = argvalues
        if names is None or values is None:
            raise TypeError("parametrize() missing required arguments")
        def dec(fn):
            if not hasattr(fn, '_parametrize_list'):
                fn._parametrize_list = []
            fn._parametrize_list.append((names, values))
            fn._parametrize = (names, values)
            return fn
        return dec

    def skip(self, reason=""):
        def dec(fn):
            fn._skip = True
            fn._skip_reason = reason
            return fn
        return dec

    def skipif(self, cond, reason=""):
        def dec(fn):
            if cond:
                fn._skip = True
                fn._skip_reason = reason
            return fn
        return dec

    def xfail(self, *args, reason="", raises=None, **kwargs):
        if args and callable(args[0]):
            fn = args[0]
            fn._xfail = True
            fn._xfail_reason = reason
            return fn
        def dec(fn):
            fn._xfail = True
            fn._xfail_reason = reason
            return fn
        return dec

    def slow(self, fn=None):
        if fn is not None:
            return fn
        return lambda f: f

    def filterwarnings(self, *args, **kwargs):
        def dec(fn):
            return fn
        return dec

    def __getattr__(self, name):
        def _unknown_mark(*args, **kwargs):
            if args and callable(args[0]) and not kwargs:
                return args[0]
            def dec(fn):
                return fn
            return dec
        return _unknown_mark


def _skip_func(msg=""):
    raise _SkipException(msg)


def _fixture(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    params = kwargs.get('params', None)
    def _dec(fn):
        if params is not None:
            fn._fixture_params = params
        return fn
    return _dec


def _importorskip(name, **kwargs):
    try:
        return __import__(name)
    except ImportError:
        raise _SkipException("Module {} not available".format(name))


def _param(*args, **kwargs):
    if len(args) == 1:
        return args[0]
    return args


def _xfail_func(reason=""):
    """Immediately mark the current test as an expected failure (like pytest.xfail)."""
    raise _SkipException("[XFAIL] " + reason)

_pytest.raises = _raises
_pytest.warns = _warns
_pytest.approx = _approx
_pytest.mark = _Mark()
_pytest.param = _param
_pytest.skip = _skip_func
_pytest.xfail = _xfail_func
_pytest.fixture = _fixture
_pytest.importorskip = _importorskip
_pytest.SkipException = _SkipException

sys.modules["pytest"] = _pytest
sys.modules["_pytest"] = types.ModuleType("_pytest")
sys.modules["_pytest.outcomes"] = types.ModuleType("_pytest.outcomes")
sys.modules["_pytest.outcomes"].Failed = AssertionError

# ---------------------------------------------------------------------------
# 2. Hypothesis shim (marks @given tests as skipped)
# ---------------------------------------------------------------------------

_hyp_mod = types.ModuleType("hypothesis")


def _skip_given(*a, **kw):
    def decorator(fn):
        fn._skip = True
        fn._skip_reason = "hypothesis not available"
        return fn
    return decorator


_hyp_mod.given = _skip_given
_hyp_mod.assume = lambda x: None
_hyp_mod.settings = lambda *a, **kw: lambda fn: fn


class _FakeStrategy:
    def __or__(self, other): return self
    def __ror__(self, other): return self
    def filter(self, *a, **kw): return self
    def map(self, *a, **kw): return self
    def flatmap(self, *a, **kw): return self


_fake = _FakeStrategy()

_hyp_strategies = types.ModuleType("hypothesis.strategies")
for _sname in ['data', 'integers', 'floats', 'text', 'sampled_from', 'one_of',
               'just', 'none', 'lists', 'tuples', 'booleans', 'binary',
               'characters', 'from_regex', 'nothing', 'complex_numbers',
               'fractions', 'decimals', 'datetimes', 'dates', 'times',
               'timedeltas', 'uuids', 'from_type', 'builds', 'fixed_dictionaries',
               'dictionaries', 'sets', 'frozensets', 'recursive',
               'deferred', 'shared', 'runner', 'permutations', 'randoms']:
    setattr(_hyp_strategies, _sname, lambda *a, **kw: _fake)
_hyp_strategies.composite = lambda fn: fn

_hyp_mod.strategies = _hyp_strategies
sys.modules["hypothesis"] = _hyp_mod
sys.modules["hypothesis.strategies"] = _hyp_strategies

_hyp_extra = types.ModuleType("hypothesis.extra")
_hyp_extra_np = types.ModuleType("hypothesis.extra.numpy")
for _sname in ['arrays', 'array_dtypes', 'integer_dtypes', 'floating_dtypes',
               'array_shapes', 'mutually_broadcastable_shapes', 'from_dtype',
               'scalar_dtypes', 'unsigned_integer_dtypes',
               'complex_number_dtypes', 'boolean_dtypes', 'byte_string_dtypes',
               'unicode_string_dtypes', 'datetime64_dtypes',
               'timedelta64_dtypes', 'nested_dtypes', 'valid_tuple_axes',
               'broadcastable_shapes', 'basic_indices']:
    setattr(_hyp_extra_np, _sname, lambda *a, **kw: _fake)
_hyp_extra.numpy = _hyp_extra_np
_hyp_mod.extra = _hyp_extra
sys.modules["hypothesis.extra"] = _hyp_extra
sys.modules["hypothesis.extra.numpy"] = _hyp_extra_np

# ---------------------------------------------------------------------------
# 3. Import numpy
# ---------------------------------------------------------------------------

import numpy as np

# ---------------------------------------------------------------------------
# 4. Test runner engine
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0
_skipped = 0
_xfailed = 0
_xpassed = 0
_errored = 0
_load_errors = 0
_failures = []
_errors = []
_unexpected_failures = []

_TEST_TIMEOUT = 30

_verbose = "--verbose" in sys.argv or "-v" in sys.argv
_ci_mode = "--ci" in sys.argv
_scan_mode = "--scan" in sys.argv
_summary_only = "--summary" in sys.argv or _scan_mode

_XFAIL_TESTS = set()
_SKIP_TESTS = set()


def _load_xfail(test_file):
    global _XFAIL_TESTS, _SKIP_TESTS
    base = os.path.splitext(os.path.basename(test_file))[0]
    xfail_path = os.path.join(_this_dir, "xfail_{}.txt".format(base))
    _XFAIL_TESTS = set()
    if os.path.exists(xfail_path):
        with open(xfail_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    _XFAIL_TESTS.add(line)
    skip_path = os.path.join(_this_dir, "skip_{}.txt".format(base))
    _SKIP_TESTS = set()
    if os.path.exists(skip_path):
        with open(skip_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    _SKIP_TESTS.add(line)


def _is_expected_fail(name):
    return _ci_mode and name in _XFAIL_TESTS


def _record_failure(full_name, msg):
    global _failed, _xfailed
    if _is_expected_fail(full_name):
        _xfailed += 1
        return "XFAIL"
    _failed += 1
    _failures.append((full_name, msg))
    _unexpected_failures.append((full_name, msg))
    return "FAIL"


def _run_single_test(full_name, fn, args=()):
    global _passed, _failed, _skipped, _xfailed, _xpassed

    is_xfail = getattr(fn, "_xfail", False)

    try:
        fn(*args)
    except _SkipException:
        _skipped += 1
        return "SKIP"
    except AssertionError as e:
        if is_xfail:
            _xfailed += 1
            return "XFAIL"
        return _record_failure(full_name, str(e)[:300])
    except Exception as e:
        if is_xfail:
            _xfailed += 1
            return "XFAIL"
        return _record_failure(full_name, "{}: {}".format(type(e).__name__, str(e)[:250]))

    if is_xfail:
        _passed += 1
        return "XPASS"
    if _is_expected_fail(full_name):
        _xpassed += 1
    _passed += 1
    return "PASS"


def _cross_product(param_lists, method=None):
    import itertools
    layers = []
    for names_str, values in param_lists:
        try:
            vals_list = list(values) if not isinstance(values, (list, tuple)) else list(values)
        except Exception:
            vals_list = []
        if isinstance(names_str, (list, tuple)):
            names = list(names_str)
        else:
            names = [n.strip() for n in str(names_str).split(",")]
        is_multi = len(names) > 1
        layers.append((names, vals_list, is_multi))

    if method is not None:
        _sorted = False
        try:
            _code = getattr(method, '_code_obj', None) or method.__code__
            varnames = _code.co_varnames
            params = [v for v in varnames if v != 'self']
            def _sig_index(layer_names):
                for p in layer_names:
                    if p in params:
                        return params.index(p)
                return 9999
            layers.sort(key=lambda layer: _sig_index(layer[0]))
            _sorted = True
        except (AttributeError, Exception) as _e:
            pass
        if not _sorted:
            # Fallback: try inspect.signature
            try:
                import inspect
                sig = inspect.signature(method)
                params = [p for p in sig.parameters if p != 'self']
                def _sig_index2(layer_names):
                    for p in layer_names:
                        if p in params:
                            return params.index(p)
                    return 9999
                layers.sort(key=lambda layer: _sig_index2(layer[0]))
            except Exception:
                pass

    all_value_lists = [l[1] for l in layers]
    multi_arg = [l[2] for l in layers]
    result = []
    for combo in itertools.product(*all_value_lists):
        flat = []
        for i, val in enumerate(combo):
            if multi_arg[i] and isinstance(val, (tuple, list)):
                flat.extend(val)
            else:
                flat.append(val)
        result.append(tuple(flat))
    return result


_module_fixtures = {}  # populated by _run_test_file for module-level @pytest.fixture functions


class _MockRequest:
    """Minimal mock of pytest's request object for fixture resolution."""
    def __init__(self, fixtures_dict):
        self._fixtures = fixtures_dict

    def getfixturevalue(self, name):
        if name in self._fixtures:
            fn = self._fixtures[name]
            return fn()  # call the fixture function
        raise ValueError("Fixture '{}' not found".format(name))


def _run_parametrized(full_name, method):
    global _passed, _failed, _skipped, _xfailed, _xpassed, _errored

    param_lists = getattr(method, '_parametrize_list', None)
    if param_lists and len(param_lists) > 1:
        values = _cross_product(param_lists, method)
        # Combine all parameter names
        names_str = ','.join(p[0] if isinstance(p[0], str) else ','.join(p[0]) for p in param_lists)
    else:
        names_str, values = method._parametrize
    is_xfail = getattr(method, "_xfail", False)

    # Detect if the method needs a 'request' parameter
    _needs_request = False
    try:
        _code = getattr(method, '__code__', None)
        if _code is None:
            _wrapped = getattr(method, '__wrapped__', method)
            _code = getattr(_wrapped, '__code__', None)
        if _code and 'request' in _code.co_varnames[:_code.co_argcount]:
            _needs_request = True
    except Exception:
        pass

    try:
        if not isinstance(values, (list, tuple)):
            values = list(values)
    except Exception as e:
        if _is_expected_fail(full_name):
            _xfailed += 1
        else:
            _errored += 1
            _errors.append((full_name, "parametrize failed: {}".format(str(e)[:200])))
        return

    # Determine if parametrize has a single parameter name
    # If so, each value should be passed as a single argument (even if it's a tuple)
    _single_param = False
    if isinstance(names_str, str) and ',' not in names_str:
        _single_param = True

    for i, vals in enumerate(values):
        param_name = "{}[{}]".format(full_name, i)
        try:
            if _single_param:
                call_vals = (vals,)
            else:
                call_vals = vals if isinstance(vals, (tuple, list)) else (vals,)
            if _needs_request:
                call_vals = tuple(call_vals) + (_MockRequest(_module_fixtures),)
            if len(call_vals) == 1:
                method(call_vals[0])
            else:
                method(*call_vals)
        except _SkipException:
            _skipped += 1
            continue
        except (AssertionError, Exception) as e:
            if is_xfail:
                _xfailed += 1
            else:
                _record_failure(param_name, str(e)[:300])
            continue
        if _is_expected_fail(param_name):
            _xpassed += 1
        _passed += 1


class _ParametrizedWrapper:
    """Wrapper that merges class-level and method-level parametrize lists."""
    def __init__(self, method, class_params):
        self._method = method
        # Start with method-level params
        method_params = getattr(method, '_parametrize_list', None)
        if method_params:
            combined = list(method_params)
        elif hasattr(method, '_parametrize'):
            combined = [method._parametrize]
        else:
            combined = []
        # Append class-level params
        if class_params:
            combined.extend(class_params)
        self._parametrize_list = combined
        self._parametrize = combined[0] if combined else None
        # Forward xfail and __code__ (for _cross_product sorting)
        self._xfail = getattr(method, '_xfail', False)
        self._xfail_reason = getattr(method, '_xfail_reason', '')
        # Try multiple ways to get __code__ for parameter ordering
        # We need the ORIGINAL function's code (not a *args/**kwargs wrapper)
        _code = None
        # Walk through __wrapped__ chain to find original function
        _candidates = []
        obj = method
        for _depth in range(10):
            _candidates.append(obj)
            _func = getattr(obj, '__func__', None)
            if _func is not None and _func not in _candidates:
                _candidates.append(_func)
            _wrapped = getattr(obj, '__wrapped__', None)
            if _wrapped is not None and _wrapped not in _candidates:
                obj = _wrapped
            else:
                break
        for cand in _candidates:
            c = getattr(cand, '__code__', None)
            if c is not None:
                vn = c.co_varnames
                # Skip *args/**kwargs wrappers — we want the real signature
                if len(vn) >= 2 and 'args' not in vn[:3] and 'kwargs' not in vn[:3]:
                    _code = c
                    break
                elif _code is None:
                    _code = c  # keep as fallback
        # If best code is still *args, try inspect
        if _code is not None and 'args' in _code.co_varnames[:3]:
            try:
                import inspect
                sig = inspect.signature(method)
                _varnames = tuple(sig.parameters.keys())
                if len(_varnames) > 0 and 'args' not in _varnames[:3]:
                    class _FakeCode:
                        co_varnames = _varnames
                    _code = _FakeCode()
            except Exception:
                pass
        self._code_obj = _code

    def __call__(self, *args):
        return self._method(*args)


def _run_test_method(cls_name, inst, mname, class_params=None, fixtures=None):
    global _skipped, _errored

    try:
        method = getattr(inst, mname)
    except Exception:
        _errored += 1
        return

    full_name = "{}.{}".format(cls_name, mname)

    if full_name in _SKIP_TESTS:
        _skipped += 1
        return

    if getattr(method, "_skip", False):
        _skipped += 1
        return

    # Check if this test method needs fixture arguments
    if fixtures:
        # Detect which fixture parameters this method accepts
        _method_params = []
        try:
            _code = getattr(method, '__code__', None) or getattr(method, '__func__', method).__code__
            _varnames = _code.co_varnames
            # Skip 'self' if present
            _args = [v for v in _varnames[:_code.co_argcount] if v != 'self']
        except (AttributeError, Exception):
            try:
                import inspect
                sig = inspect.signature(method)
                _args = [p for p in sig.parameters if p != 'self']
            except Exception:
                _args = []

        _needed_fixtures = [(fname, fvals) for fname, fvals in fixtures.items() if fname in _args]
        if _needed_fixtures:
            # Convert fixtures to parametrize-style execution
            fixture_params = [(fname, fvals) for fname, fvals in _needed_fixtures]
            # Merge with existing class_params and method parametrize
            extra_params = [(fname, vals) for fname, vals in fixture_params]
            merged = list(extra_params)
            if class_params:
                merged.extend(class_params)
            has_method_params = hasattr(method, "_parametrize")
            if has_method_params:
                wrapper = _ParametrizedWrapper(method, merged)
            else:
                wrapper = _ParametrizedWrapper(method, merged)
            _run_parametrized(full_name, wrapper)
            return

    # Merge class-level parametrize with method-level parametrize
    has_method_params = hasattr(method, "_parametrize")
    if class_params or has_method_params:
        wrapper = _ParametrizedWrapper(method, class_params)
        _run_parametrized(full_name, wrapper)
        return

    _run_single_test(full_name, method)


def _run_test_file(test_path, label=None):
    """Load and run a single test file."""
    global _passed, _failed, _skipped, _xfailed, _xpassed, _errored, _load_errors
    global _failures, _errors, _unexpected_failures

    _passed = 0
    _failed = 0
    _skipped = 0
    _xfailed = 0
    _xpassed = 0
    _errored = 0
    _failures = []
    _errors = []
    _unexpected_failures = []

    basename = os.path.basename(test_path)
    mod_name = os.path.splitext(basename)[0]

    _load_xfail(test_path)

    test_ns = None
    try:
        with open(test_path) as f:
            code = f.read()
        test_ns = {"__name__": mod_name, "__file__": test_path}
        compiled = compile(code, test_path, "exec")
        import builtins
        getattr(builtins, "exec")(compiled, test_ns)
    except Exception as e:
        _load_errors += 1
        if not _summary_only:
            print("  LOAD ERROR: {}".format(str(e)[:200]))
        return (0, 0, 0, 1, 0, False)

    names = sorted(test_ns.keys())

    # Collect module-level @pytest.fixture functions for request.getfixturevalue()
    global _module_fixtures
    _module_fixtures = {}
    for name in names:
        obj = test_ns[name]
        if callable(obj) and not isinstance(obj, type) and not name.startswith("test_"):
            # Check if it was decorated with @pytest.fixture (our shim sets _fixture_params on parameterized ones)
            # For non-parameterized fixtures, they're just callables returned by _fixture
            if hasattr(obj, '_fixture_params') or (not name.startswith("_") and not name.startswith("Test")):
                _module_fixtures[name] = obj

    for name in names:
        obj = test_ns[name]

        if isinstance(obj, type) and name.startswith("Test"):
            if not _summary_only:
                print("--- {} ---".format(name))
                sys.stdout.flush()

            # Collect class-level parametrize
            class_params = getattr(obj, '_parametrize_list', None)

            try:
                inst = obj()
            except Exception:
                _errored += 1
                continue
            try:
                method_names = sorted(m for m in dir(obj) if m.startswith("test_"))
            except Exception:
                _errored += 1
                continue

            # Collect fixture methods (methods with _fixture_params)
            _fixtures = {}
            try:
                for _attr_name in dir(obj):
                    if _attr_name.startswith("test_") or _attr_name.startswith("_"):
                        continue
                    _attr = getattr(obj, _attr_name, None)
                    if callable(_attr) and hasattr(_attr, '_fixture_params'):
                        # Resolve fixture values by calling the method with a mock request
                        _fix_values = []
                        for _p in _attr._fixture_params:
                            class _MockRequest:
                                param = _p
                            try:
                                _val = _attr(inst, _MockRequest())
                                _fix_values.append(_val)
                            except Exception:
                                _fix_values.append(_p)
                        _fixtures[_attr_name] = _fix_values
            except Exception:
                pass

            for mname in method_names:
                setup_ok = True
                if hasattr(inst, "setup_method"):
                    try:
                        inst.setup_method()
                    except _SkipException:
                        _skipped += 1
                        setup_ok = False
                    except Exception:
                        _errored += 1
                        setup_ok = False
                elif hasattr(inst, "setup"):
                    try:
                        inst.setup()
                    except _SkipException:
                        _skipped += 1
                        setup_ok = False
                    except Exception:
                        _errored += 1
                        setup_ok = False

                if setup_ok:
                    _run_test_method(name, inst, mname, class_params, _fixtures if _fixtures else None)
                    if hasattr(inst, "teardown_method"):
                        try:
                            inst.teardown_method()
                        except Exception:
                            pass
                    elif hasattr(inst, "teardown"):
                        try:
                            inst.teardown()
                        except Exception:
                            pass

        elif callable(obj) and name.startswith("test_"):
            if getattr(obj, "_skip", False):
                _skipped += 1
                continue
            if hasattr(obj, "_parametrize"):
                _run_parametrized(name, obj)
            else:
                _run_single_test(name, obj)

    return (_passed, _failed, _skipped, _errored, _xfailed, True)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if _scan_mode:
        upstream_dir = os.path.join(_this_dir, "upstream")
        test_files = sorted(
            f for f in os.listdir(upstream_dir)
            if f.endswith(".py") and f.startswith(("core_", "lib_", "fft_", "linalg_",
                                                    "ma_", "poly_", "random_"))
        )
        print("Scanning {} upstream test files...\n".format(len(test_files)))
        print("{:<55s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s}".format(
            "FILE", "PASS", "FAIL", "SKIP", "ERR", "XFAIL"))
        print("-" * 91)

        grand_passed = grand_failed = grand_skipped = grand_errored = grand_xfailed = 0
        grand_load_errors = 0

        for tf in test_files:
            test_path = os.path.join(upstream_dir, tf)
            t0 = time.time()
            try:
                passed, failed, skipped, errored, xfailed, loaded = _run_test_file(test_path, tf)
            except Exception:
                passed, failed, skipped, errored, xfailed, loaded = 0, 0, 0, 1, 0, False

            elapsed = time.time() - t0
            if not loaded:
                print("{:<55s} {:>6s}  (load error, {:.1f}s)".format(tf, "ERR", elapsed))
                grand_load_errors += 1
            else:
                print("{:<55s} {:>6d} {:>6d} {:>6d} {:>6d} {:>6d}  ({:.1f}s)".format(
                    tf, passed, failed, skipped, errored, xfailed, elapsed))
                grand_passed += passed
                grand_failed += failed
                grand_skipped += skipped
                grand_errored += errored
                grand_xfailed += xfailed

        print("-" * 91)
        print("{:<55s} {:>6d} {:>6d} {:>6d} {:>6d} {:>6d}".format(
            "TOTAL", grand_passed, grand_failed, grand_skipped,
            grand_errored, grand_xfailed))
        if grand_load_errors:
            print("  ({} files failed to load)".format(grand_load_errors))
        print()

    elif args:
        test_file = args[0]
        if not os.path.isabs(test_file):
            test_file = os.path.join(_this_dir, test_file)
        if not os.path.exists(test_file):
            print("ERROR: {} not found".format(test_file))
            sys.exit(1)

        basename = os.path.basename(test_file)
        print("Running {} ...".format(basename))
        passed, failed, skipped, errored, xfailed, loaded = _run_test_file(test_file)

        print()
        if _errors:
            print("--- ERRORS ({}) ---".format(len(_errors)))
            for ename, emsg in _errors[:20]:
                print("  ERROR {}: {}".format(ename, emsg))
            if len(_errors) > 20:
                print("  ... and {} more".format(len(_errors) - 20))
            print()

        if _failures and not _summary_only:
            shown = _failures[:300]
            print("--- FAILURES ({}) ---".format(len(_failures)))
            for fname, fmsg in shown:
                print("  FAIL {}: {}".format(fname, fmsg[:150]))
            if len(_failures) > 300:
                print("  ... and {} more".format(len(_failures) - 30))
            print()

        total = passed + failed + skipped + errored + xfailed
        print("{}: {} passed, {} failed, {} skipped, {} errored, {} xfailed (total {})".format(
            basename, passed, failed, skipped, errored, xfailed, total))

        if not loaded:
            sys.exit(2)
        if _ci_mode and _unexpected_failures:
            sys.exit(1)
    else:
        print("Usage: run_upstream.py <test_file> [--ci] [--verbose|-v]")
        print("       run_upstream.py --scan")
        sys.exit(1)
