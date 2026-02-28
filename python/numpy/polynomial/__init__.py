"""numpy.polynomial - polynomial module."""


class polynomial:
    """numpy.polynomial.polynomial module.

    Provides polynomial functions using ascending coefficient order:
    c[0] + c[1]*x + c[2]*x^2 + ...

    This is REVERSED from numpy.polyval which uses descending order.
    """

    @staticmethod
    def polyval(x, c):
        """Evaluate polynomial with coefficients c at points x.
        Coefficient order: c[0] + c[1]*x + c[2]*x^2 + ...
        """
        import numpy as np
        x = np.asarray(x)
        c_list = list(np.asarray(c).tolist())
        result = np.zeros(x.shape) if hasattr(x, 'shape') else 0.0
        for i, ci in enumerate(c_list):
            result = result + float(ci) * np.power(x, i)
        return result

    @staticmethod
    def polyfit(x, y, deg):
        """Fit polynomial of degree deg. Returns coefficients in ascending order."""
        import numpy as np
        coeffs = np.polyfit(x, y, deg)
        c_list = coeffs.tolist()
        c_list.reverse()
        return np.array(c_list)

    @staticmethod
    def polyadd(c1, c2):
        """Add two polynomials (ascending coefficient order)."""
        import numpy as np
        c1 = list(np.asarray(c1).tolist())
        c2 = list(np.asarray(c2).tolist())
        n = max(len(c1), len(c2))
        while len(c1) < n:
            c1.append(0.0)
        while len(c2) < n:
            c2.append(0.0)
        return np.array([c1[i] + c2[i] for i in range(n)])

    @staticmethod
    def polysub(c1, c2):
        """Subtract two polynomials (ascending coefficient order)."""
        import numpy as np
        c1 = list(np.asarray(c1).tolist())
        c2 = list(np.asarray(c2).tolist())
        n = max(len(c1), len(c2))
        while len(c1) < n:
            c1.append(0.0)
        while len(c2) < n:
            c2.append(0.0)
        return np.array([c1[i] - c2[i] for i in range(n)])

    @staticmethod
    def polymul(c1, c2):
        """Multiply two polynomials (ascending coefficient order)."""
        import numpy as np
        c1 = list(np.asarray(c1).tolist())
        c2 = list(np.asarray(c2).tolist())
        n = len(c1) + len(c2) - 1
        result = [0.0] * n
        for i in range(len(c1)):
            for j in range(len(c2)):
                result[i + j] += c1[i] * c2[j]
        return np.array(result)

    @staticmethod
    def polyder(c, m=1):
        """Differentiate a polynomial (ascending coefficient order)."""
        import numpy as np
        c = list(np.asarray(c).tolist())
        for _ in range(m):
            c = [c[i] * i for i in range(1, len(c))]
        return np.array(c) if c else np.array([0.0])

    @staticmethod
    def polyint(c, m=1):
        """Integrate a polynomial (ascending coefficient order)."""
        import numpy as np
        c = list(np.asarray(c).tolist())
        for _ in range(m):
            c = [0.0] + [c[i] / (i + 1) for i in range(len(c))]
        return np.array(c)


# Also expose as module-level
Polynomial = polynomial
