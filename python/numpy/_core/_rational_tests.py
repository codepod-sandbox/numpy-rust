"""numpy._core._rational_tests stub."""


class rational:
    """Stub for the rational dtype used in NumPy C tests."""
    def __init__(self, num=0, den=1):
        self.numerator = num
        self.denominator = den

    def __repr__(self):
        return f"rational({self.numerator}, {self.denominator})"
