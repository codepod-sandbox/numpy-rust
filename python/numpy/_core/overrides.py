"""numpy._core.overrides - array function dispatch mechanism."""


def set_module(module):
    """Decorator to set __module__ on a function."""
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator


def array_function_dispatch(dispatcher=None, module=None, verify=True,
                            docs_from_dispatcher=False):
    """No-op decorator for array function dispatch."""
    def decorator(implementation):
        if module is not None:
            implementation.__module__ = module
        return implementation
    return decorator


def array_function_from_dispatcher(implementation, module=None, verify=True,
                                    docs_from_dispatcher=False):
    """No-op passthrough."""
    return implementation


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
