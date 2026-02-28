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


class chebyshev:
    """Chebyshev polynomial operations.

    Chebyshev polynomials of the first kind T_n(x):
      T_0(x) = 1
      T_1(x) = x
      T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)

    Coefficients are in ascending order: c[0]*T_0 + c[1]*T_1 + c[2]*T_2 + ...
    """

    @staticmethod
    def chebval(x, c):
        """Use Clenshaw recurrence to evaluate Chebyshev series at points x."""
        import numpy as np
        x = np.asarray(x)
        c_list = list(c) if isinstance(c, list) else list(np.asarray(c).flatten().tolist())
        if len(c_list) == 0:
            return np.zeros_like(x) if hasattr(x, 'shape') else 0.0
        if len(c_list) == 1:
            return np.full(x.shape, c_list[0]) if hasattr(x, 'shape') and len(x.shape) > 0 else float(c_list[0])

        nd = len(c_list)
        c0 = float(c_list[-2])
        c1 = float(c_list[-1])
        x2 = x * 2.0
        for i in range(3, nd + 1):
            tmp = c0
            c0 = float(c_list[-i]) - c1
            c1 = tmp + c1 * x2
        return c0 + c1 * x

    @staticmethod
    def chebfit(x, y, deg):
        """Least squares fit of Chebyshev series to data."""
        import numpy as np
        x = np.asarray(x)
        y = np.asarray(y)
        x_list = x.flatten().tolist()
        n = deg + 1
        vander = []
        for xi in x_list:
            row = [0.0] * n
            row[0] = 1.0
            if n > 1:
                row[1] = xi
            for j in range(2, n):
                row[j] = 2.0 * xi * row[j - 1] - row[j - 2]
            vander.append(row)
        V = np.array(vander)
        coeffs = np.linalg.lstsq(V, y)[0]
        return coeffs

    @staticmethod
    def chebadd(c1, c2):
        """Add two Chebyshev series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        maxlen = max(len(c1), len(c2))
        while len(c1) < maxlen:
            c1.append(0.0)
        while len(c2) < maxlen:
            c2.append(0.0)
        return np.array([a + b for a, b in zip(c1, c2)])

    @staticmethod
    def chebsub(c1, c2):
        """Subtract two Chebyshev series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        maxlen = max(len(c1), len(c2))
        while len(c1) < maxlen:
            c1.append(0.0)
        while len(c2) < maxlen:
            c2.append(0.0)
        return np.array([a - b for a, b in zip(c1, c2)])

    @staticmethod
    def chebmul(c1, c2):
        """Multiply two Chebyshev series.

        Uses the identity: T_m * T_n = 0.5*(T_{m+n} + T_{|m-n|})
        """
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        n1 = len(c1)
        n2 = len(c2)
        out_len = n1 + n2 - 1
        result = [0.0] * out_len
        for i in range(n1):
            for j in range(n2):
                v = c1[i] * c2[j]
                idx_sum = i + j
                idx_diff = abs(i - j)
                if i == 0 or j == 0:
                    result[idx_sum] += v if (i == 0 and j == 0) else v
                else:
                    result[idx_sum] += v * 0.5
                    result[idx_diff] += v * 0.5
        return np.array(result)

    @staticmethod
    def chebder(c, m=1):
        """Differentiate a Chebyshev series.

        Uses the backward recurrence:
          dc[n-2] = 2*(n-1)*c[n-1]
          dc[k] = dc[k+2] + 2*(k+1)*c[k+1]  for k = n-3, ..., 0
          dc[0] *= 0.5
        """
        import numpy as np
        c = list(np.asarray(c).flatten().tolist())
        for _ in range(m):
            n = len(c)
            if n <= 1:
                c = [0.0]
                continue
            dc = [0.0] * (n - 1)
            dc[n - 2] = 2.0 * (n - 1) * c[n - 1]
            if n - 3 >= 0:
                dc[n - 3] = 2.0 * (n - 2) * c[n - 2]
            for k in range(n - 4, -1, -1):
                dc[k] = dc[k + 2] + 2.0 * (k + 1) * c[k + 1]
            dc[0] *= 0.5
            c = dc
        return np.array(c)

    @staticmethod
    def chebint(c, m=1):
        """Integrate a Chebyshev series.

        Uses: integral(T_n) = T_{n+1}/(2*(n+1)) - T_{n-1}/(2*(n-1)) for n >= 2
        integral(T_0) = T_1
        integral(T_1) = T_2/4 + T_0*... (special handling)
        """
        import numpy as np
        c = list(np.asarray(c).flatten().tolist())
        for _ in range(m):
            n = len(c)
            if n == 0:
                c = [0.0]
                continue
            ic = [0.0] * (n + 1)
            for j in range(n):
                if j == 0:
                    ic[1] += c[0]
                elif j == 1:
                    ic[2] += c[1] / 4.0
                else:
                    ic[j + 1] += c[j] / (2.0 * (j + 1))
                    ic[j - 1] -= c[j] / (2.0 * (j - 1))
            c = ic
        return np.array(c)


class legendre:
    """Legendre polynomial operations.

    Legendre polynomials P_n(x):
      P_0(x) = 1
      P_1(x) = x
      (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)

    Coefficients are in ascending order: c[0]*P_0 + c[1]*P_1 + c[2]*P_2 + ...
    """

    @staticmethod
    def legval(x, c):
        """Compute Legendre series at points x."""
        import numpy as np
        x = np.asarray(x)
        c_list = list(np.asarray(c).flatten().tolist())
        if len(c_list) == 0:
            return np.zeros_like(x) if hasattr(x, 'shape') else 0.0
        if len(c_list) == 1:
            return np.full(x.shape, c_list[0]) if hasattr(x, 'shape') and len(x.shape) > 0 else float(c_list[0])
        nd = len(c_list)
        p0 = np.ones(x.shape) if hasattr(x, 'shape') else 1.0
        p1 = x * 1.0
        result = c_list[0] * p0 + c_list[1] * p1
        for i in range(2, nd):
            p2 = ((2.0 * i - 1.0) * x * p1 - (i - 1.0) * p0) / float(i)
            result = result + c_list[i] * p2
            p0 = p1
            p1 = p2
        return result

    @staticmethod
    def legfit(x, y, deg):
        """Least squares fit of Legendre series to data."""
        import numpy as np
        x = np.asarray(x)
        y = np.asarray(y)
        x_list = x.flatten().tolist()
        n = deg + 1
        vander = []
        for xi in x_list:
            row = [0.0] * n
            row[0] = 1.0
            if n > 1:
                row[1] = xi
            for j in range(2, n):
                row[j] = ((2 * j - 1) * xi * row[j - 1] - (j - 1) * row[j - 2]) / j
            vander.append(row)
        V = np.array(vander)
        return np.linalg.lstsq(V, y)[0]

    @staticmethod
    def legadd(c1, c2):
        """Add two Legendre series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        maxlen = max(len(c1), len(c2))
        while len(c1) < maxlen:
            c1.append(0.0)
        while len(c2) < maxlen:
            c2.append(0.0)
        return np.array([a + b for a, b in zip(c1, c2)])

    @staticmethod
    def legsub(c1, c2):
        """Subtract two Legendre series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        maxlen = max(len(c1), len(c2))
        while len(c1) < maxlen:
            c1.append(0.0)
        while len(c2) < maxlen:
            c2.append(0.0)
        return np.array([a - b for a, b in zip(c1, c2)])

    @staticmethod
    def legmul(c1, c2):
        """Multiply two Legendre series using linearization."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        n1 = len(c1)
        n2 = len(c2)
        out_deg = n1 + n2 - 2
        n_pts = out_deg + 1
        import math
        pts = [math.cos(math.pi * (2 * k + 1) / (2 * n_pts)) for k in range(n_pts)]
        vals = []
        for xi in pts:
            v1 = 0.0
            p0, p1 = 1.0, xi
            v1 += c1[0] * p0
            if n1 > 1:
                v1 += c1[1] * p1
            for j in range(2, n1):
                p2 = ((2 * j - 1) * xi * p1 - (j - 1) * p0) / j
                v1 += c1[j] * p2
                p0, p1 = p1, p2

            v2 = 0.0
            p0, p1 = 1.0, xi
            v2 += c2[0] * p0
            if n2 > 1:
                v2 += c2[1] * p1
            for j in range(2, n2):
                p2 = ((2 * j - 1) * xi * p1 - (j - 1) * p0) / j
                v2 += c2[j] * p2
                p0, p1 = p1, p2
            vals.append(v1 * v2)
        x_arr = np.array(pts)
        y_arr = np.array(vals)
        return legendre.legfit(x_arr, y_arr, out_deg)

    @staticmethod
    def legder(c, m=1):
        """Differentiate a Legendre series.

        Uses the backward recurrence for Legendre derivative coefficients.
        """
        import numpy as np
        c = list(np.asarray(c).flatten().tolist())
        for _ in range(m):
            n = len(c)
            if n <= 1:
                c = [0.0]
                continue
            dc = [0.0] * (n - 1)
            for j in range(n - 1, 0, -1):
                dc[j - 1] += (2 * j - 1) * c[j]
                if j - 2 >= 0:
                    c[j - 2] += c[j]
            c = dc
        return np.array(c)

    @staticmethod
    def legint(c, m=1):
        """Integrate a Legendre series.

        Uses: integral(P_n) = (P_{n+1} - P_{n-1})/(2n+1) for n >= 1
        integral(P_0) = P_1
        """
        import numpy as np
        c = list(np.asarray(c).flatten().tolist())
        for _ in range(m):
            n = len(c)
            if n == 0:
                c = [0.0]
                continue
            ic = [0.0] * (n + 1)
            for j in range(n):
                if j == 0:
                    ic[1] += c[0]
                else:
                    ic[j + 1] += c[j] / (2.0 * j + 1.0)
                    if j - 1 >= 0:
                        ic[j - 1] -= c[j] / (2.0 * j + 1.0)
            c = ic
        return np.array(c)


class hermite:
    """Hermite polynomial operations (physicist's convention H_n).

    Physicist's Hermite polynomials:
      H_0(x) = 1
      H_1(x) = 2x
      H_{n+1}(x) = 2x*H_n(x) - 2n*H_{n-1}(x)

    Coefficients are in ascending order: c[0]*H_0 + c[1]*H_1 + c[2]*H_2 + ...
    """

    @staticmethod
    def hermval(x, c):
        """Compute physicist's Hermite series at points x."""
        import numpy as np
        x = np.asarray(x)
        c_list = list(np.asarray(c).flatten().tolist())
        if len(c_list) == 0:
            return np.zeros_like(x) if hasattr(x, 'shape') else 0.0
        if len(c_list) == 1:
            return np.full(x.shape, c_list[0]) if hasattr(x, 'shape') and len(x.shape) > 0 else float(c_list[0])
        nd = len(c_list)
        h0 = np.ones(x.shape) if hasattr(x, 'shape') else 1.0
        h1 = x * 2.0
        result = c_list[0] * h0 + c_list[1] * h1
        for i in range(2, nd):
            h2 = 2.0 * x * h1 - 2.0 * (i - 1) * h0
            result = result + c_list[i] * h2
            h0 = h1
            h1 = h2
        return result

    @staticmethod
    def hermfit(x, y, deg):
        """Least squares fit of physicist's Hermite series to data."""
        import numpy as np
        x = np.asarray(x)
        y = np.asarray(y)
        x_list = x.flatten().tolist()
        n = deg + 1
        vander = []
        for xi in x_list:
            row = [0.0] * n
            row[0] = 1.0
            if n > 1:
                row[1] = 2.0 * xi
            for j in range(2, n):
                row[j] = 2.0 * xi * row[j - 1] - 2.0 * (j - 1) * row[j - 2]
            vander.append(row)
        V = np.array(vander)
        return np.linalg.lstsq(V, y)[0]

    @staticmethod
    def hermadd(c1, c2):
        """Add two Hermite series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        maxlen = max(len(c1), len(c2))
        while len(c1) < maxlen:
            c1.append(0.0)
        while len(c2) < maxlen:
            c2.append(0.0)
        return np.array([a + b for a, b in zip(c1, c2)])

    @staticmethod
    def hermsub(c1, c2):
        """Subtract two Hermite series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        maxlen = max(len(c1), len(c2))
        while len(c1) < maxlen:
            c1.append(0.0)
        while len(c2) < maxlen:
            c2.append(0.0)
        return np.array([a - b for a, b in zip(c1, c2)])

    @staticmethod
    def hermmul(c1, c2):
        """Multiply two Hermite series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        n1 = len(c1)
        n2 = len(c2)
        out_deg = n1 + n2 - 2
        n_pts = out_deg + 1
        pts = [2.0 * k / max(n_pts - 1, 1) - 1.0 for k in range(n_pts)]
        vals = []
        for xi in pts:
            v1 = 0.0
            h0, h1 = 1.0, 2.0 * xi
            v1 += c1[0] * h0
            if n1 > 1:
                v1 += c1[1] * h1
            for j in range(2, n1):
                h2 = 2.0 * xi * h1 - 2.0 * (j - 1) * h0
                v1 += c1[j] * h2
                h0, h1 = h1, h2
            v2 = 0.0
            h0, h1 = 1.0, 2.0 * xi
            v2 += c2[0] * h0
            if n2 > 1:
                v2 += c2[1] * h1
            for j in range(2, n2):
                h2 = 2.0 * xi * h1 - 2.0 * (j - 1) * h0
                v2 += c2[j] * h2
                h0, h1 = h1, h2
            vals.append(v1 * v2)
        x_arr = np.array(pts)
        y_arr = np.array(vals)
        return hermite.hermfit(x_arr, y_arr, out_deg)

    @staticmethod
    def hermder(c, m=1):
        """Differentiate a physicist's Hermite series.

        Uses: H'_n(x) = 2n * H_{n-1}(x)
        """
        import numpy as np
        c = list(np.asarray(c).flatten().tolist())
        for _ in range(m):
            n = len(c)
            if n <= 1:
                c = [0.0]
                continue
            dc = [0.0] * (n - 1)
            for j in range(1, n):
                dc[j - 1] = 2.0 * j * c[j]
            c = dc
        return np.array(c)

    @staticmethod
    def hermint(c, m=1):
        """Integrate a physicist's Hermite series.

        Uses: integral of H_n = H_{n+1} / (2*(n+1))
        """
        import numpy as np
        c = list(np.asarray(c).flatten().tolist())
        for _ in range(m):
            n = len(c)
            ic = [0.0] * (n + 1)
            for j in range(n):
                ic[j + 1] = c[j] / (2.0 * (j + 1))
            c = ic
        return np.array(c)


class hermite_e:
    """Hermite polynomial operations (probabilist's convention He_n).

    Probabilist's Hermite polynomials:
      He_0(x) = 1
      He_1(x) = x
      He_{n+1}(x) = x*He_n(x) - n*He_{n-1}(x)

    Coefficients are in ascending order: c[0]*He_0 + c[1]*He_1 + c[2]*He_2 + ...
    """

    @staticmethod
    def hermeval(x, c):
        """Compute probabilist's Hermite series at points x."""
        import numpy as np
        x = np.asarray(x)
        c_list = list(np.asarray(c).flatten().tolist())
        if len(c_list) == 0:
            return np.zeros_like(x) if hasattr(x, 'shape') else 0.0
        if len(c_list) == 1:
            return np.full(x.shape, c_list[0]) if hasattr(x, 'shape') and len(x.shape) > 0 else float(c_list[0])
        nd = len(c_list)
        h0 = np.ones(x.shape) if hasattr(x, 'shape') else 1.0
        h1 = x * 1.0
        result = c_list[0] * h0 + c_list[1] * h1
        for i in range(2, nd):
            h2 = x * h1 - (i - 1) * h0
            result = result + c_list[i] * h2
            h0 = h1
            h1 = h2
        return result

    @staticmethod
    def hermefit(x, y, deg):
        """Least squares fit of probabilist's Hermite series to data."""
        import numpy as np
        x = np.asarray(x)
        y = np.asarray(y)
        x_list = x.flatten().tolist()
        n = deg + 1
        vander = []
        for xi in x_list:
            row = [0.0] * n
            row[0] = 1.0
            if n > 1:
                row[1] = xi
            for j in range(2, n):
                row[j] = xi * row[j - 1] - (j - 1) * row[j - 2]
            vander.append(row)
        V = np.array(vander)
        return np.linalg.lstsq(V, y)[0]

    @staticmethod
    def hermeadd(c1, c2):
        """Add two probabilist's Hermite series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        maxlen = max(len(c1), len(c2))
        while len(c1) < maxlen:
            c1.append(0.0)
        while len(c2) < maxlen:
            c2.append(0.0)
        return np.array([a + b for a, b in zip(c1, c2)])

    @staticmethod
    def hermesub(c1, c2):
        """Subtract two probabilist's Hermite series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        maxlen = max(len(c1), len(c2))
        while len(c1) < maxlen:
            c1.append(0.0)
        while len(c2) < maxlen:
            c2.append(0.0)
        return np.array([a - b for a, b in zip(c1, c2)])

    @staticmethod
    def hermeder(c, m=1):
        """Differentiate a probabilist's Hermite series.

        Uses: He'_n(x) = n * He_{n-1}(x)
        """
        import numpy as np
        c = list(np.asarray(c).flatten().tolist())
        for _ in range(m):
            n = len(c)
            if n <= 1:
                c = [0.0]
                continue
            dc = [0.0] * (n - 1)
            for j in range(1, n):
                dc[j - 1] = float(j) * c[j]
            c = dc
        return np.array(c)

    @staticmethod
    def hermeint(c, m=1):
        """Integrate a probabilist's Hermite series.

        Uses: integral of He_n = He_{n+1} / (n+1)
        """
        import numpy as np
        c = list(np.asarray(c).flatten().tolist())
        for _ in range(m):
            n = len(c)
            ic = [0.0] * (n + 1)
            for j in range(n):
                ic[j + 1] = c[j] / float(j + 1)
            c = ic
        return np.array(c)


class laguerre:
    """Laguerre polynomial operations.

    Laguerre polynomials L_n(x):
      L_0(x) = 1
      L_1(x) = 1 - x
      (n+1)*L_{n+1}(x) = (2n+1-x)*L_n(x) - n*L_{n-1}(x)

    Coefficients are in ascending order: c[0]*L_0 + c[1]*L_1 + c[2]*L_2 + ...
    """

    @staticmethod
    def lagval(x, c):
        """Compute Laguerre series at points x."""
        import numpy as np
        x = np.asarray(x)
        c_list = list(np.asarray(c).flatten().tolist())
        if len(c_list) == 0:
            return np.zeros_like(x) if hasattr(x, 'shape') else 0.0
        if len(c_list) == 1:
            return np.full(x.shape, c_list[0]) if hasattr(x, 'shape') and len(x.shape) > 0 else float(c_list[0])
        nd = len(c_list)
        l0 = np.ones(x.shape) if hasattr(x, 'shape') else 1.0
        l1 = (np.ones(x.shape) if hasattr(x, 'shape') else 1.0) - x
        result = c_list[0] * l0 + c_list[1] * l1
        for i in range(2, nd):
            l2 = ((2.0 * i - 1.0 - x) * l1 - (i - 1.0) * l0) / float(i)
            result = result + c_list[i] * l2
            l0 = l1
            l1 = l2
        return result

    @staticmethod
    def lagfit(x, y, deg):
        """Least squares fit of Laguerre series to data."""
        import numpy as np
        x = np.asarray(x)
        y = np.asarray(y)
        x_list = x.flatten().tolist()
        n = deg + 1
        vander = []
        for xi in x_list:
            row = [0.0] * n
            row[0] = 1.0
            if n > 1:
                row[1] = 1.0 - xi
            for j in range(2, n):
                row[j] = ((2 * j - 1 - xi) * row[j - 1] - (j - 1) * row[j - 2]) / j
            vander.append(row)
        V = np.array(vander)
        return np.linalg.lstsq(V, y)[0]

    @staticmethod
    def lagadd(c1, c2):
        """Add two Laguerre series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        maxlen = max(len(c1), len(c2))
        while len(c1) < maxlen:
            c1.append(0.0)
        while len(c2) < maxlen:
            c2.append(0.0)
        return np.array([a + b for a, b in zip(c1, c2)])

    @staticmethod
    def lagsub(c1, c2):
        """Subtract two Laguerre series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        maxlen = max(len(c1), len(c2))
        while len(c1) < maxlen:
            c1.append(0.0)
        while len(c2) < maxlen:
            c2.append(0.0)
        return np.array([a - b for a, b in zip(c1, c2)])

    @staticmethod
    def lagmul(c1, c2):
        """Multiply two Laguerre series."""
        import numpy as np
        c1 = list(np.asarray(c1).flatten().tolist())
        c2 = list(np.asarray(c2).flatten().tolist())
        n1 = len(c1)
        n2 = len(c2)
        out_deg = n1 + n2 - 2
        n_pts = out_deg + 1
        pts = [float(k) * 10.0 / max(n_pts - 1, 1) for k in range(n_pts)]
        vals = []
        for xi in pts:
            v1 = 0.0
            l0_v, l1_v = 1.0, 1.0 - xi
            v1 += c1[0] * l0_v
            if n1 > 1:
                v1 += c1[1] * l1_v
            for j in range(2, n1):
                l2_v = ((2 * j - 1 - xi) * l1_v - (j - 1) * l0_v) / j
                v1 += c1[j] * l2_v
                l0_v, l1_v = l1_v, l2_v
            v2 = 0.0
            l0_v, l1_v = 1.0, 1.0 - xi
            v2 += c2[0] * l0_v
            if n2 > 1:
                v2 += c2[1] * l1_v
            for j in range(2, n2):
                l2_v = ((2 * j - 1 - xi) * l1_v - (j - 1) * l0_v) / j
                v2 += c2[j] * l2_v
                l0_v, l1_v = l1_v, l2_v
            vals.append(v1 * v2)
        x_arr = np.array(pts)
        y_arr = np.array(vals)
        return laguerre.lagfit(x_arr, y_arr, out_deg)

    @staticmethod
    def lagder(c, m=1):
        """Differentiate a Laguerre series.

        Uses: L'_n(x) = -sum_{k=0}^{n-1} L_k(x) for n >= 1
        So d/dx (sum c_j L_j) = -sum_k L_k * (sum_{j=k+1}^{n-1} c_j)
        """
        import numpy as np
        c = list(np.asarray(c).flatten().tolist())
        for _ in range(m):
            n = len(c)
            if n <= 1:
                c = [0.0]
                continue
            dc = [0.0] * (n - 1)
            for k in range(n - 1):
                s = 0.0
                for j in range(k + 1, n):
                    s -= c[j]
                dc[k] = s
            c = dc
        return np.array(c)

    @staticmethod
    def lagint(c, m=1):
        """Integrate a Laguerre series.

        Uses: integral of L_n = L_n - L_{n+1}
        """
        import numpy as np
        c = list(np.asarray(c).flatten().tolist())
        for _ in range(m):
            n = len(c)
            if n == 0:
                c = [0.0]
                continue
            ic = [0.0] * (n + 1)
            for j in range(n):
                ic[j] += c[j]
                ic[j + 1] -= c[j]
            c = ic
        return np.array(c)


# Expose class-level aliases
Chebyshev = chebyshev
Legendre = legendre
Hermite = hermite
HermiteE = hermite_e
Laguerre = laguerre
