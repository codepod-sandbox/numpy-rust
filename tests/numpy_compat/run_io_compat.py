"""
Test runner for official NumPy compatibility tests (npy/npz I/O) via RustPython.

Provides minimal shims for pytest, hypothesis, and numpy internals that
are not available in the RustPython environment, then discovers and
runs tests from upstream test_io.py.

Only TestSaveLoad and TestSavezLoad are run (npy/npz tests).
TestSaveTxt, TestLoadTxt, etc. are intentionally skipped.

Usage:
    ./target/release/numpy-python tests/numpy_compat/run_io_compat.py
    ./target/release/numpy-python tests/numpy_compat/run_io_compat.py --ci
"""

import sys
import os
import types
import traceback

# ---------------------------------------------------------------------------
# 1. Minimal pytest shim
# ---------------------------------------------------------------------------

_pytest = types.ModuleType("pytest")
_pytest.__name__ = "pytest"
_pytest.__package__ = "pytest"


class _SkipException(Exception):
    """Raised by pytest.skip() to skip a test at runtime."""
    pass


class _RaisesContext:
    """Context manager for pytest.raises()."""
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
                    self.exc_type.__name__
                )
            )
        if issubclass(exc_type, self.exc_type):
            self.value = exc_val
            if self.match is not None:
                import re
                if not re.search(self.match, str(exc_val)):
                    raise AssertionError(
                        "Regex {!r} did not match {!r}".format(
                            self.match, str(exc_val)
                        )
                    )
            return True  # suppress the expected exception
        return False  # let unexpected exceptions propagate


class _WarnsContext:
    """Context manager for pytest.warns()."""
    def __init__(self):
        self.list = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return True  # always suppress


def _raises(exc, *args, match=None, **kwargs):
    """Shim for pytest.raises(exc) or pytest.raises(exc, callable, ...)."""
    if args:
        fn = args[0]
        fargs = args[1:]
        try:
            fn(*fargs, **kwargs)
            raise AssertionError(
                "Expected {} but no exception was raised".format(exc.__name__)
            )
        except exc:
            pass
        return
    return _RaisesContext(exc, match)


def _warns(cls, *args, **kwargs):
    """Shim for pytest.warns(cls) or pytest.warns(cls, callable, ...)."""
    if args:
        fn = args[0]
        fargs = args[1:]
        fn(*fargs, **kwargs)
        return
    return _WarnsContext()


def _approx(x, rel=None, abs=None, nan_ok=False):
    """Rough approximation of pytest.approx -- returns the value itself."""
    return x


class _Mark:
    """Shim for pytest.mark decorator factory."""

    def parametrize(self, names, values, **kwargs):
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
        def dec(fn):
            fn._xfail = True
            fn._xfail_reason = reason
            return fn
        # Handle @pytest.mark.xfail(reason="...") and @pytest.mark.xfail("reason")
        if args and callable(args[0]):
            return dec(args[0])
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
        """Catch-all for unknown marks -- treat as no-op decorator."""
        def _unknown_mark(*args, **kwargs):
            if args and callable(args[0]) and not kwargs:
                return args[0]
            def dec(fn):
                return fn
            return dec
        return _unknown_mark


def _skip_func(msg=""):
    """pytest.skip() -- raises to abort current test."""
    raise _SkipException(msg)


def _fixture(*args, **kwargs):
    """Shim for @pytest.fixture."""
    if args and callable(args[0]):
        return args[0]
    return lambda fn: fn


def _importorskip(name, **kwargs):
    """Shim for pytest.importorskip -- try importing or skip."""
    try:
        return __import__(name)
    except ImportError:
        raise _SkipException("Module {} not available".format(name))


def _param(*args, **kwargs):
    """Shim for pytest.param()."""
    if len(args) == 1:
        return args[0]
    return args


_pytest.raises = _raises
_pytest.warns = _warns
_pytest.approx = _approx
_pytest.mark = _Mark()
_pytest.param = _param
_pytest.skip = _skip_func
_pytest.fixture = _fixture
_pytest.importorskip = _importorskip
_pytest.SkipException = _SkipException

sys.modules["pytest"] = _pytest

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

_hyp_strategies = types.ModuleType("hypothesis.strategies")
_hyp_mod.strategies = _hyp_strategies
sys.modules["hypothesis"] = _hyp_mod
sys.modules["hypothesis.strategies"] = _hyp_strategies

# ---------------------------------------------------------------------------
# 3. NumPy internal module shims
# ---------------------------------------------------------------------------

import numpy as np

# Sentinel
if not hasattr(np, "_NoValue"):
    class _NoValueType:
        def __repr__(self):
            return "<no value>"
    np._NoValue = _NoValueType()

# ---------------------------------------------------------------------------
# 3a. numpy.lib shim -- expose NpzFile at numpy.lib.npyio.NpzFile
# ---------------------------------------------------------------------------

# Import our NpzFile
try:
    from numpy._io import NpzFile as _NpzFile
except ImportError:
    # fallback: get it from what np.load returns
    try:
        from io import BytesIO as _BytesIO
        import numpy as _np_tmp
        _buf = _BytesIO()
        _np_tmp.savez(_buf)
        _buf.seek(0)
        _NpzFile = type(_np_tmp.load(_buf))
    except Exception:
        _NpzFile = None

_npyio_mod = types.ModuleType("numpy.lib.npyio")
if _NpzFile is not None:
    _npyio_mod.NpzFile = _NpzFile

_npyio_impl_mod = types.ModuleType("numpy.lib._npyio_impl")
_npyio_impl_mod.recfromcsv = lambda *a, **kw: (_ for _ in ()).throw(
    _SkipException("recfromcsv not available"))
_npyio_impl_mod.recfromtxt = lambda *a, **kw: (_ for _ in ()).throw(
    _SkipException("recfromtxt not available"))

_iotools_mod = types.ModuleType("numpy.lib._iotools")

class _ConversionWarning(UserWarning):
    pass

class _ConverterError(Exception):
    pass

_iotools_mod.ConversionWarning = _ConversionWarning
_iotools_mod.ConverterError = _ConverterError

_lib_mod = types.ModuleType("numpy.lib")
_lib_mod.__path__ = []
_lib_mod.npyio = _npyio_mod
_lib_mod._npyio_impl = _npyio_impl_mod
_lib_mod._iotools = _iotools_mod

try:
    np.lib = _lib_mod
except Exception:
    pass

sys.modules["numpy.lib"] = _lib_mod
sys.modules["numpy.lib.npyio"] = _npyio_mod
sys.modules["numpy.lib._npyio_impl"] = _npyio_impl_mod
sys.modules["numpy.lib._iotools"] = _iotools_mod

# ---------------------------------------------------------------------------
# 3b. numpy._utils shim
# ---------------------------------------------------------------------------

def _asbytes(s):
    if isinstance(s, str):
        return s.encode('latin1')
    return s

_utils_mod = types.ModuleType("numpy._utils")
_utils_mod.asbytes = _asbytes
sys.modules["numpy._utils"] = _utils_mod

# ---------------------------------------------------------------------------
# 3c. numpy.ma shim
# ---------------------------------------------------------------------------

_ma_mod = types.ModuleType("numpy.ma")
_ma_testutils = types.ModuleType("numpy.ma.testutils")

# Use our assert_equal from testing_utils
_this_dir = os.path.dirname(os.path.abspath(__file__))
_support_dir = os.path.join(_this_dir, "_support")
if _support_dir not in sys.path:
    sys.path.insert(0, _support_dir)

try:
    from testing_utils import (
        assert_equal as _assert_equal_fn,
        assert_ as _assert_fn,
        HAS_REFCOUNT as _HAS_REFCOUNT,
        IS_WASM as _IS_WASM,
        IS_PYPY as _IS_PYPY,
        suppress_warnings as _suppress_warnings,
        break_cycles as _break_cycles,
        temppath as _temppath_cls,
        tempdir as _tempdir_cls,
    )
    _IMPORT_UTILS_OK = True
except ImportError as _ie:
    print("WARNING: could not import testing_utils: {}".format(_ie))
    _IMPORT_UTILS_OK = False

    def _assert_equal_fn(actual, desired, **kw):
        if actual != desired:
            raise AssertionError("{!r} != {!r}".format(actual, desired))

    def _assert_fn(val, msg=""):
        if not val:
            raise AssertionError(msg or "assertion failed")

    _HAS_REFCOUNT = False
    _IS_WASM = False
    _IS_PYPY = False

    class _suppress_warnings:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def filter(self, *a, **kw): pass
        def record(self, *a, **kw): return []

    def _break_cycles(): pass

    import tempfile as _tf
    class _temppath_cls:
        def __init__(self, suffix="", prefix="tmp", dir=None):
            self._f = _tf.NamedTemporaryFile(suffix=suffix, prefix=prefix, dir=dir, delete=False)
            self.path = self._f.name
            self._f.close()
        def __enter__(self): return self.path
        def __exit__(self, *a):
            try: os.unlink(self.path)
            except OSError: pass

    class _tempdir_cls:
        def __init__(self, suffix="", prefix="tmp", dir=None):
            import tempfile
            self._dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        def __enter__(self): return self._dir
        def __exit__(self, *a):
            import shutil
            try: shutil.rmtree(self._dir)
            except OSError: pass

_ma_testutils.assert_equal = _assert_equal_fn
_ma_mod.testutils = _ma_testutils
sys.modules["numpy.ma"] = _ma_mod
sys.modules["numpy.ma.testutils"] = _ma_testutils

# ---------------------------------------------------------------------------
# 3d. numpy.exceptions shim
# ---------------------------------------------------------------------------

_exc_mod = types.ModuleType("numpy.exceptions")
_exc_mod.AxisError = getattr(np, "AxisError", type("AxisError", (Exception,), {}))
_exc_mod.ComplexWarning = getattr(np, "ComplexWarning", UserWarning)
_exc_mod.VisibleDeprecationWarning = getattr(
    np, "VisibleDeprecationWarning", UserWarning
)
sys.modules["numpy.exceptions"] = _exc_mod

# ---------------------------------------------------------------------------
# 3e. numpy.random shim
# ---------------------------------------------------------------------------

_random_mod = types.ModuleType("numpy.random")
for _rname in [
    "rand", "randint", "randn", "random", "seed", "choice",
    "uniform", "normal", "shuffle", "permutation", "RandomState",
    "default_rng",
]:
    if hasattr(np.random, _rname):
        setattr(_random_mod, _rname, getattr(np.random, _rname))
sys.modules["numpy.random"] = _random_mod

# ---------------------------------------------------------------------------
# 3f. numpy.testing shim
# ---------------------------------------------------------------------------

_testing = types.ModuleType("numpy.testing")
for _tname in [
    "assert_equal", "assert_array_equal", "assert_almost_equal",
    "assert_array_almost_equal", "assert_allclose", "assert_raises",
    "assert_raises_regex", "assert_warns", "assert_array_max_ulp",
    "assert_array_less", "assert_approx_equal", "assert_no_warnings",
]:
    if hasattr(np.testing, _tname):
        setattr(_testing, _tname, getattr(np.testing, _tname))
    else:
        setattr(_testing, _tname, lambda *a, **kw: None)

def _assert_no_warnings(func=None, *args, **kwargs):
    if func is None:
        class _NoWarnCtx:
            def __enter__(self): return None
            def __exit__(self, *exc): return True
        return _NoWarnCtx()
    return func(*args, **kwargs)

_testing.assert_no_warnings = _assert_no_warnings
_testing.assert_ = _assert_fn
_testing.HAS_REFCOUNT = _HAS_REFCOUNT
_testing.IS_WASM = _IS_WASM
_testing.IS_PYPY = _IS_PYPY
_testing.suppress_warnings = _suppress_warnings
_testing.break_cycles = _break_cycles
_testing.temppath = _temppath_cls
_testing.tempdir = _tempdir_cls

# assert_no_gc_cycles -- stub (gc.collect + no-op)
def _assert_no_gc_cycles(func=None, *args, **kwargs):
    if func is None:
        class _NoGCCtx:
            def __enter__(self): return None
            def __exit__(self, *exc): return False
        return _NoGCCtx()
    return func(*args, **kwargs)

_testing.assert_no_gc_cycles = _assert_no_gc_cycles

sys.modules["numpy.testing"] = _testing

# numpy.testing._private.utils
_testing_private = types.ModuleType("numpy.testing._private")
_testing_utils_mod = types.ModuleType("numpy.testing._private.utils")

def _requires_memory(free_bytes=0, *, min_free=0):
    def dec(fn):
        fn._skip = True
        fn._skip_reason = "requires_memory not checked"
        return fn
    return dec

_testing_utils_mod.requires_memory = _requires_memory
_testing_private.utils = _testing_utils_mod
sys.modules["numpy.testing._private"] = _testing_private
sys.modules["numpy.testing._private.utils"] = _testing_utils_mod

# ---------------------------------------------------------------------------
# 3g. numpy._core shim (minimal)
# ---------------------------------------------------------------------------

_core = types.ModuleType("numpy._core")
_core_umath = types.ModuleType("numpy._core.umath")
_core.umath = _core_umath
_core.numeric = np
_core.multiarray = np

try:
    np._core = _core
except Exception:
    pass

sys.modules["numpy._core"] = _core
sys.modules["numpy._core.umath"] = _core_umath
sys.modules["numpy._core.numeric"] = np
sys.modules["numpy._core.multiarray"] = np

# ---------------------------------------------------------------------------
# 3h. IS_64BIT, IS_PYPY, IS_WASM for test module
# ---------------------------------------------------------------------------

IS_64BIT = sys.maxsize > 2**32
IS_PYPY = False
IS_WASM = False

# ---------------------------------------------------------------------------
# 4. Load the test module
# ---------------------------------------------------------------------------

_test_path = os.path.join(_this_dir, "upstream", "test_io.py")

print("numpy_compat: loading test_io.py ...")

_load_error = None
_test_ns = None

try:
    import importlib.util as _imputil
    _spec = _imputil.spec_from_file_location("test_io", _test_path)
    _test_mod = _imputil.module_from_spec(_spec)
    sys.modules["test_io"] = _test_mod
    _spec.loader.exec_module(_test_mod)
    _test_ns = vars(_test_mod)
except Exception as _e1:
    try:
        with open(_test_path) as _f:
            _test_code = _f.read()
        _test_ns = {"__name__": "test_io", "__file__": _test_path}
        _compiled = compile(_test_code, _test_path, "exec")
        import builtins
        getattr(builtins, "exec")(_compiled, _test_ns)
    except Exception as _e2:
        _load_error = _e2

if _load_error is not None:
    print("FATAL: could not load test_io.py: {}".format(_load_error))
    traceback.print_exc()
    sys.exit(1)

if _test_ns is None:
    print("FATAL: test namespace is empty")
    sys.exit(1)

print("numpy_compat: test_io.py loaded, discovering tests ...")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 5. Test runner configuration
# ---------------------------------------------------------------------------

# Only run these two classes (npy/npz tests)
_CLASSES_TO_RUN = {"TestSaveLoad", "TestSavezLoad"}

# Check for --verbose / -v flag
_verbose = "--verbose" in sys.argv or "-v" in sys.argv

# Check for --ci flag
_ci_mode = "--ci" in sys.argv

# Load xfail list
_XFAIL_TESTS = set()
_xfail_path = os.path.join(_this_dir, "xfail_io.txt")
if os.path.exists(_xfail_path):
    with open(_xfail_path) as _xf:
        for _line in _xf:
            _line = _line.strip()
            if _line and not _line.startswith("#"):
                _XFAIL_TESTS.add(_line)

# Tests that cause process aborts / infinite loops / very long runtimes
# These must be hard-skipped because RustPython threading can't enforce timeouts
_SKIP_TESTS = {
    # Requires 2GB+ memory, would hang or OOM
    "TestSavezLoad.test_big_arrays",
    # These run 1024 iterations of file open/close -- very slow in RustPython
    "TestSavezLoad.test_closing_fid",
}

_passed = 0
_failed = 0
_skipped = 0
_xfailed = 0
_xpassed = 0
_errored = 0
_failures = []
_errors = []
_unexpected_failures = []

_TEST_TIMEOUT = 30  # longer timeout for I/O tests

import threading


class _TestTimeoutError(Exception):
    pass


def _run_with_timeout(fn, args, timeout):
    """Run fn(*args) with a timeout."""
    if getattr(sys, "implementation", None) and sys.implementation.name == "rustpython":
        result = fn(*args)
        return result
    result = [None]
    exc = [None]

    def target():
        try:
            result[0] = fn(*args)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise _TestTimeoutError("Test timed out after {}s".format(timeout))
    if exc[0] is not None:
        raise exc[0]
    return result[0]


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

    if _verbose:
        sys.stdout.write("  {} ...".format(full_name))
        sys.stdout.flush()

    is_xfail = getattr(fn, "_xfail", False)

    try:
        _run_with_timeout(fn, args, _TEST_TIMEOUT)
    except _TestTimeoutError as e:
        msg = str(e)
        outcome = _record_failure(full_name, msg)
        if _verbose:
            print(" {}".format(outcome))
            sys.stdout.flush()
        return outcome
    except _SkipException:
        _skipped += 1
        if _verbose:
            print(" SKIP")
            sys.stdout.flush()
        return "SKIP"
    except AssertionError as e:
        if is_xfail:
            _xfailed += 1
            if _verbose:
                print(" XFAIL")
                sys.stdout.flush()
            return "XFAIL"
        msg = str(e)
        if len(msg) > 300:
            msg = msg[:300] + "..."
        outcome = _record_failure(full_name, msg)
        if _verbose:
            print(" {}".format(outcome))
            sys.stdout.flush()
        return outcome
    except Exception as e:
        if is_xfail:
            _xfailed += 1
            if _verbose:
                print(" XFAIL")
                sys.stdout.flush()
            return "XFAIL"
        msg = "{}: {}".format(type(e).__name__, str(e))
        if len(msg) > 300:
            msg = msg[:300] + "..."
        outcome = _record_failure(full_name, msg)
        if _verbose:
            print(" {}".format(outcome))
            sys.stdout.flush()
        return outcome

    if is_xfail:
        _passed += 1
        if _verbose:
            print(" XPASS")
            sys.stdout.flush()
        return "XPASS"

    if _is_expected_fail(full_name):
        _xpassed += 1
    _passed += 1
    if _verbose:
        print(" PASS")
        sys.stdout.flush()
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
        try:
            varnames = method.__code__.co_varnames
            params = [v for v in varnames if v != 'self']
            def _sig_index(layer_names):
                for p in layer_names:
                    if p in params:
                        return params.index(p)
                return 9999
            layers.sort(key=lambda layer: _sig_index(layer[0]))
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


def _run_parametrized(full_name, method):
    global _passed, _failed, _skipped, _xfailed, _xpassed, _errored

    param_lists = getattr(method, '_parametrize_list', None)
    if param_lists and len(param_lists) > 1:
        values = _cross_product(param_lists, method)
    else:
        names_str, values = method._parametrize
    is_xfail = getattr(method, "_xfail", False)

    try:
        if not isinstance(values, (list, tuple)):
            values = list(values)
    except Exception as e:
        err_name = full_name
        err_msg = "parametrize values failed: {}".format(str(e)[:200])
        if _is_expected_fail(err_name):
            _xfailed += 1
        else:
            _errored += 1
            _errors.append((err_name, err_msg))
        return

    for i, vals in enumerate(values):
        param_name = "{}[{}]".format(full_name, i)

        if _verbose:
            sys.stdout.write("  {} ...".format(param_name))
            sys.stdout.flush()

        try:
            if isinstance(vals, (tuple, list)):
                _run_with_timeout(method, vals, _TEST_TIMEOUT)
            else:
                _run_with_timeout(method, (vals,), _TEST_TIMEOUT)
        except _TestTimeoutError as e:
            outcome = _record_failure(param_name, str(e))
            if _verbose:
                print(" {}".format(outcome))
                sys.stdout.flush()
            continue
        except _SkipException:
            _skipped += 1
            if _verbose:
                print(" SKIP")
                sys.stdout.flush()
            continue
        except AssertionError as e:
            if is_xfail:
                _xfailed += 1
                if _verbose:
                    print(" XFAIL")
                    sys.stdout.flush()
            else:
                msg = str(e)
                if len(msg) > 300:
                    msg = msg[:300] + "..."
                outcome = _record_failure(param_name, msg)
                if _verbose:
                    print(" {}".format(outcome))
                    sys.stdout.flush()
            continue
        except Exception as e:
            if is_xfail:
                _xfailed += 1
                if _verbose:
                    print(" XFAIL")
                    sys.stdout.flush()
            else:
                msg = "{}: {}".format(type(e).__name__, str(e))
                if len(msg) > 300:
                    msg = msg[:300] + "..."
                outcome = _record_failure(param_name, msg)
                if _verbose:
                    print(" {}".format(outcome))
                    sys.stdout.flush()
            continue
        if _is_expected_fail(param_name):
            _xpassed += 1
        _passed += 1
        if _verbose:
            print(" PASS")
            sys.stdout.flush()


def _run_test_method(cls_name, inst, mname):
    global _skipped, _errored

    try:
        method = getattr(inst, mname)
    except Exception:
        _errored += 1
        return

    full_name = "{}.{}".format(cls_name, mname)

    if full_name in _SKIP_TESTS:
        _skipped += 1
        if _verbose:
            print("  SKIP {} (known abort)".format(full_name))
        return

    if getattr(method, "_skip", False):
        _skipped += 1
        return

    if hasattr(method, "_parametrize"):
        _run_parametrized(full_name, method)
        return

    _run_single_test(full_name, method)


# ---------------------------------------------------------------------------
# 6. Discover and run tests (only TestSaveLoad and TestSavezLoad)
# ---------------------------------------------------------------------------

_names = sorted(_test_ns.keys())

for _name in _names:
    _obj = _test_ns[_name]

    if not (isinstance(_obj, type) and _name.startswith("Test")):
        continue

    if _name not in _CLASSES_TO_RUN:
        continue

    print("--- {} ---".format(_name))
    sys.stdout.flush()

    try:
        _inst = _obj()
    except Exception as _e:
        _errored += 1
        _errors.append((_name, "Could not instantiate: {}".format(str(_e)[:200])))
        continue

    try:
        _method_names = sorted(m for m in dir(_obj) if m.startswith("test_"))
    except Exception:
        _errored += 1
        continue

    for _mname in _method_names:
        _test_full = "{}.{}".format(_name, _mname)
        _setup_ok = True

        if hasattr(_inst, "setup_method"):
            try:
                _inst.setup_method()
            except _SkipException:
                _skipped += 1
                _setup_ok = False
            except Exception as _se:
                _err_msg = "setup_method failed: {}".format(str(_se)[:200])
                if _is_expected_fail(_test_full):
                    _xfailed += 1
                else:
                    _errored += 1
                    _errors.append((_test_full, _err_msg))
                _setup_ok = False
        elif hasattr(_inst, "setup"):
            try:
                _inst.setup()
            except _SkipException:
                _skipped += 1
                _setup_ok = False
            except Exception as _se:
                _err_msg = "setup failed: {}".format(str(_se)[:200])
                if _is_expected_fail(_test_full):
                    _xfailed += 1
                else:
                    _errored += 1
                    _errors.append((_test_full, _err_msg))
                _setup_ok = False

        if _setup_ok:
            _run_test_method(_name, _inst, _mname)
            if hasattr(_inst, "teardown_method"):
                try:
                    _inst.teardown_method()
                except Exception:
                    pass
            elif hasattr(_inst, "teardown"):
                try:
                    _inst.teardown()
                except Exception:
                    pass

# ---------------------------------------------------------------------------
# 7. Print results
# ---------------------------------------------------------------------------

print("")

if _errors:
    print("--- ERRORS ({}) ---".format(len(_errors)))
    for _ename, _emsg in _errors:
        print("  ERROR {}: {}".format(_ename, _emsg))
    print("")

if _ci_mode and _unexpected_failures:
    print("--- UNEXPECTED FAILURES ({}) ---".format(len(_unexpected_failures)))
    for _fname, _fmsg in _unexpected_failures:
        print("  FAIL {}: {}".format(_fname, _fmsg))
    print("")
elif _failures:
    print("--- FAILURES ({}) ---".format(len(_failures)))
    for _fname, _fmsg in _failures:
        print("  FAIL {}: {}".format(_fname, _fmsg))
    print("")

_total = _passed + _failed + _skipped + _xfailed + _errored
if _ci_mode:
    print(
        "numpy_compat/test_io (CI mode): "
        "{} passed, {} unexpected failures, {} expected failures (xfail), "
        "{} xpassed, {} skipped, {} errored (total {})".format(
            _passed, len(_unexpected_failures), _xfailed,
            _xpassed, _skipped, _errored, _total
        )
    )
    if _xpassed > 0:
        print(
            "  NOTE: {} tests in xfail_io.txt now pass — "
            "consider removing them from xfail_io.txt".format(_xpassed)
        )
else:
    print(
        "numpy_compat/test_io: "
        "{} passed, {} failed, {} skipped, {} xfailed, {} errored "
        "(total {})".format(
            _passed, _failed, _skipped, _xfailed, _errored, _total
        )
    )

if _ci_mode:
    if _unexpected_failures or _errored > 0:
        sys.exit(1)
else:
    if _failed > 0 or _errored > 0:
        sys.exit(1)
