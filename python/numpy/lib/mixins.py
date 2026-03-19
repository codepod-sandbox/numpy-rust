"""numpy.lib.mixins - mixin classes."""
import numpy as np


def _disallow_array_ufunc(obj):
    """Check if obj opts out of __array_ufunc__."""
    return getattr(type(obj), '__array_ufunc__', NotImplemented) is None


def _binary_method(ufunc, name):
    """Generate a binary operator method that delegates to __array_ufunc__."""
    def func(self, other):
        if _disallow_array_ufunc(other):
            return NotImplemented
        return self.__array_ufunc__(ufunc, '__call__', self, other)
    func.__name__ = f'__{name}__'
    return func


def _reflected_binary_method(ufunc, name):
    """Generate a reflected binary operator method."""
    def func(self, other):
        if _disallow_array_ufunc(other):
            return NotImplemented
        return self.__array_ufunc__(ufunc, '__call__', other, self)
    func.__name__ = f'__r{name}__'
    return func


def _inplace_binary_method(ufunc, name):
    """Generate an in-place binary operator method."""
    def func(self, other):
        result = self.__array_ufunc__(ufunc, '__call__', self, other, out=(self,))
        if result is NotImplemented:
            raise TypeError(
                f"unsupported operand type(s) for i{name}: "
                f"'{type(self).__name__}' and '{type(other).__name__}'")
        return result
    func.__name__ = f'__i{name}__'
    return func


def _unary_method(ufunc, name):
    """Generate a unary operator method."""
    def func(self):
        return self.__array_ufunc__(ufunc, '__call__', self)
    func.__name__ = f'__{name}__'
    return func


class NDArrayOperatorsMixin:
    """Mixin defining all operator special methods using __array_ufunc__."""

    __slots__ = ()

    # Comparison operators
    __lt__ = _binary_method(np.less, 'lt')
    __le__ = _binary_method(np.less_equal, 'le')
    __eq__ = _binary_method(np.equal, 'eq')
    __ne__ = _binary_method(np.not_equal, 'ne')
    __gt__ = _binary_method(np.greater, 'gt')
    __ge__ = _binary_method(np.greater_equal, 'ge')

    # Unary operators
    __neg__ = _unary_method(np.negative, 'neg')
    __pos__ = _unary_method(np.positive, 'pos')
    __abs__ = _unary_method(np.absolute, 'abs')
    __invert__ = _unary_method(np.invert, 'invert')

    # Arithmetic operators
    __add__ = _binary_method(np.add, 'add')
    __radd__ = _reflected_binary_method(np.add, 'add')
    __iadd__ = _inplace_binary_method(np.add, 'add')
    __sub__ = _binary_method(np.subtract, 'sub')
    __rsub__ = _reflected_binary_method(np.subtract, 'sub')
    __isub__ = _inplace_binary_method(np.subtract, 'sub')
    __mul__ = _binary_method(np.multiply, 'mul')
    __rmul__ = _reflected_binary_method(np.multiply, 'mul')
    __imul__ = _inplace_binary_method(np.multiply, 'mul')
    __matmul__ = _binary_method(np.matmul, 'matmul')
    __rmatmul__ = _reflected_binary_method(np.matmul, 'matmul')
    __imatmul__ = _inplace_binary_method(np.matmul, 'matmul')
    __truediv__ = _binary_method(np.true_divide, 'truediv')
    __rtruediv__ = _reflected_binary_method(np.true_divide, 'truediv')
    __itruediv__ = _inplace_binary_method(np.true_divide, 'truediv')
    __floordiv__ = _binary_method(np.floor_divide, 'floordiv')
    __rfloordiv__ = _reflected_binary_method(np.floor_divide, 'floordiv')
    __ifloordiv__ = _inplace_binary_method(np.floor_divide, 'floordiv')
    __mod__ = _binary_method(np.remainder, 'mod')
    __rmod__ = _reflected_binary_method(np.remainder, 'mod')
    __imod__ = _inplace_binary_method(np.remainder, 'mod')
    __divmod__ = _binary_method(np.divmod, 'divmod')
    __rdivmod__ = _reflected_binary_method(np.divmod, 'divmod')
    __pow__ = _binary_method(np.power, 'pow')
    __rpow__ = _reflected_binary_method(np.power, 'pow')
    __ipow__ = _inplace_binary_method(np.power, 'pow')
    __lshift__ = _binary_method(np.left_shift, 'lshift')
    __rlshift__ = _reflected_binary_method(np.left_shift, 'lshift')
    __ilshift__ = _inplace_binary_method(np.left_shift, 'lshift')
    __rshift__ = _binary_method(np.right_shift, 'rshift')
    __rrshift__ = _reflected_binary_method(np.right_shift, 'rshift')
    __irshift__ = _inplace_binary_method(np.right_shift, 'rshift')
    __and__ = _binary_method(np.bitwise_and, 'and')
    __rand__ = _reflected_binary_method(np.bitwise_and, 'and')
    __iand__ = _inplace_binary_method(np.bitwise_and, 'and')
    __xor__ = _binary_method(np.bitwise_xor, 'xor')
    __rxor__ = _reflected_binary_method(np.bitwise_xor, 'xor')
    __ixor__ = _inplace_binary_method(np.bitwise_xor, 'xor')
    __or__ = _binary_method(np.bitwise_or, 'or')
    __ror__ = _reflected_binary_method(np.bitwise_or, 'or')
    __ior__ = _inplace_binary_method(np.bitwise_or, 'or')
