"""numpy._core.fromnumeric - array methods re-exported as functions."""
import numpy

# Re-export array manipulation/reduction functions
reshape = numpy.reshape
transpose = numpy.transpose
sort = numpy.sort
argsort = numpy.argsort
argmax = numpy.argmax
argmin = numpy.argmin
searchsorted = numpy.searchsorted
nonzero = numpy.nonzero
ravel = numpy.ravel
squeeze = numpy.squeeze
sum = numpy.sum
prod = numpy.prod
mean = numpy.mean
std = numpy.std
var = numpy.var
min = numpy.min
max = numpy.max
all = numpy.all
any = numpy.any
cumsum = numpy.cumsum
cumprod = numpy.cumprod
clip = numpy.clip
around = numpy.around


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
