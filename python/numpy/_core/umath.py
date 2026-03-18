"""numpy._core.umath - ufunc constants and re-exports."""
import numpy

PINF = float("inf")
NINF = float("-inf")
PZERO = 0.0
NZERO = -0.0

# Re-export ufuncs
add = numpy.add
subtract = numpy.subtract
multiply = numpy.multiply
true_divide = numpy.true_divide
floor_divide = numpy.floor_divide
power = numpy.power
remainder = numpy.remainder
mod = numpy.mod
absolute = numpy.abs
negative = numpy.negative
sign = numpy.sign
sqrt = numpy.sqrt
square = numpy.square
log = numpy.log
log2 = numpy.log2
log10 = numpy.log10
exp = numpy.exp
exp2 = numpy.exp2
sin = numpy.sin
cos = numpy.cos
tan = numpy.tan
arcsin = numpy.arcsin
arccos = numpy.arccos
arctan = numpy.arctan
arctan2 = numpy.arctan2
sinh = numpy.sinh
cosh = numpy.cosh
tanh = numpy.tanh
arcsinh = numpy.arcsinh
arccosh = numpy.arccosh
arctanh = numpy.arctanh
floor = numpy.floor
ceil = numpy.ceil
trunc = numpy.trunc
isnan = numpy.isnan
isinf = numpy.isinf
isfinite = numpy.isfinite
maximum = numpy.maximum
minimum = numpy.minimum
fmax = numpy.fmax
fmin = numpy.fmin
equal = numpy.equal
not_equal = numpy.not_equal
less = numpy.less
less_equal = numpy.less_equal
greater = numpy.greater
greater_equal = numpy.greater_equal
logical_and = numpy.logical_and
logical_or = numpy.logical_or
logical_not = numpy.logical_not
logical_xor = numpy.logical_xor
bitwise_and = numpy.bitwise_and
bitwise_or = numpy.bitwise_or
bitwise_xor = numpy.bitwise_xor
left_shift = numpy.left_shift
right_shift = numpy.right_shift


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
