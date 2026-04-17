"""numpy.lib._function_base_impl - function base implementation."""


def add_newdoc(place, obj, doc, warn_on_python=True):
    """Stub for add_newdoc - no-op in this implementation."""
    pass


def _lerp(a, b, t):
    return a + t * (b - a)


def _parse_gufunc_signature(signature):
    """Parse a gufunc signature into input/output core-dimension specs."""
    if not isinstance(signature, str):
        raise ValueError("gufunc signature must be a string")

    text = signature.strip()
    parts = text.split("->")
    if len(parts) != 2:
        raise ValueError("not a valid gufunc signature")
    in_text, out_text = parts[0].strip(), parts[1].strip()
    if not in_text or not out_text:
        raise ValueError("not a valid gufunc signature")

    def _parse_side(side):
        specs = []
        i = 0
        n = len(side)
        while i < n:
            while i < n and side[i].isspace():
                i += 1
            if i >= n:
                break
            if side[i] != "(":
                raise ValueError("not a valid gufunc signature")
            i += 1
            dims = []
            current = ""
            while i < n:
                ch = side[i]
                if ch == "(":
                    raise ValueError("not a valid gufunc signature")
                if ch == ")":
                    name = current.strip()
                    if name:
                        dims.append(name)
                    current = ""
                    i += 1
                    break
                if ch == ",":
                    name = current.strip()
                    if name:
                        dims.append(name)
                    elif current != "":
                        raise ValueError("not a valid gufunc signature")
                    current = ""
                    i += 1
                    continue
                current += ch
                i += 1
            else:
                raise ValueError("not a valid gufunc signature")
            specs.append(tuple(dims))
            while i < n and side[i].isspace():
                i += 1
            if i < n:
                if side[i] != ",":
                    raise ValueError("not a valid gufunc signature")
                i += 1
        if not specs:
            raise ValueError("not a valid gufunc signature")
        return specs

    return _parse_side(in_text), _parse_side(out_text)


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._function_base_impl' has no attribute {name!r}")
