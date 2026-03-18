"""numpy._utils - internal utilities."""


def asbytes(s):
    if isinstance(s, bytes):
        return s
    return s.encode('latin-1')


def asunicode(s):
    if isinstance(s, str):
        return s
    return s.decode('latin-1')
