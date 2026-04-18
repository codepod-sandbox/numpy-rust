"""numpy.strings - thin wrappers over numpy.char string operations."""
import numpy as np


def add(x1, x2):
    return np.char.add(x1, x2)


def multiply(a, i):
    return np.char.multiply(a, i)


def center(a, width, fillchar=' '):
    return np.char.center(a, width, fillchar)


def ljust(a, width, fillchar=' '):
    return np.char.ljust(a, width, fillchar)


def rjust(a, width, fillchar=' '):
    return np.char.rjust(a, width, fillchar)


def zfill(a, width):
    return np.char.zfill(a, width)


def upper(a):
    return np.char.upper(a)


def lower(a):
    return np.char.lower(a)


def swapcase(a):
    return np.char.swapcase(a)


def capitalize(a):
    return np.char.capitalize(a)


def title(a):
    return np.char.title(a)


def replace(a, old, new, count=-1):
    return np.char.replace(a, old, new, count)


def strip(a, chars=None):
    return np.char.strip(a, chars)


def lstrip(a, chars=None):
    return np.char.lstrip(a, chars)


def rstrip(a, chars=None):
    return np.char.rstrip(a, chars)


def split(a, sep=None, maxsplit=-1):
    return np.char.split(a, sep, maxsplit)


def join(sep, seq):
    return np.char.join(sep, seq)


def count(a, sub, start=0, end=None):
    return np.char.count(a, sub, start, end)


def find(a, sub, start=0, end=None):
    return np.char.find(a, sub, start, end)


def rfind(a, sub, start=0, end=None):
    return np.char.rfind(a, sub, start, end)


def index(a, sub, start=0, end=None):
    return np.char.index(a, sub, start, end)


def rindex(a, sub, start=0, end=None):
    return np.char.rindex(a, sub, start, end)


def startswith(a, prefix, start=0, end=None):
    return np.char.startswith(a, prefix, start, end)


def endswith(a, suffix, start=0, end=None):
    return np.char.endswith(a, suffix, start, end)


def encode(a, encoding='utf-8', errors='strict'):
    return np.char.encode(a, encoding, errors)


def decode(a, encoding='utf-8', errors='strict'):
    return np.char.decode(a, encoding, errors)


def isalpha(a):
    return np.char.isalpha(a)


def isdigit(a):
    return np.char.isdigit(a)


def isalnum(a):
    return np.char.isalnum(a)


def isspace(a):
    return np.char.isspace(a)


def isupper(a):
    return np.char.isupper(a)


def islower(a):
    return np.char.islower(a)


def istitle(a):
    return np.char.istitle(a)


def isnumeric(a):
    return np.char.isnumeric(a)


def isdecimal(a):
    return np.char.isdecimal(a)


def partition(a, sep):
    return np.char.partition(a, sep)


def rpartition(a, sep):
    return np.char.rpartition(a, sep)


def expandtabs(a, tabsize=8):
    return np.char.expandtabs(a, tabsize)


def mod(a, values):
    return np.char.mod(a, values)


def equal(x1, x2):
    return np.char.equal(x1, x2)


def not_equal(x1, x2):
    return np.char.not_equal(x1, x2)


def greater(x1, x2):
    return np.char.greater(x1, x2)


def greater_equal(x1, x2):
    return np.char.greater_equal(x1, x2)


def less(x1, x2):
    return np.char.less(x1, x2)


def less_equal(x1, x2):
    return np.char.less_equal(x1, x2)


def str_len(a):
    return np.char.str_len(a)


def translate(a, table, deletechars=None):
    return np.char.translate(a, table, deletechars)
