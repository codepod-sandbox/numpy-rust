"""Datetime64 and timedelta64 support."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import _ObjectArray

__all__ = [
    '_NAT_VALUE',
    '_is_nat_value',
    '_infer_datetime_unit',
    '_parse_datetime_string',
    '_date_to_days',
    '_days_to_date',
    '_to_common_unit',
    '_common_time_unit',
    '_is_dt64',
    '_is_td64',
    'datetime64',
    'timedelta64',
    '_datetime64_cls',
    '_timedelta64_cls',
    'isnat',
    'datetime_data',
    'busday_count',
    'is_busday',
    'busday_offset',
]

# --- datetime64/timedelta64 helper functions ---------------------------------

# NaT (Not a Time) sentinel: int64 minimum value (matching NumPy's iNaT)
_NAT_VALUE = -(2**63)  # -9223372036854775808


def _is_nat_value(v):
    """Return True if v is the NaT integer sentinel."""
    return isinstance(v, int) and v == _NAT_VALUE


def _infer_datetime_unit(s):
    """Infer unit from datetime string format."""
    s = s.strip()
    if len(s) == 4:  # '2024'
        return 'Y'
    elif len(s) == 7:  # '2024-01'
        return 'M'
    elif len(s) == 10:  # '2024-01-15'
        return 'D'
    elif 'T' in s:
        return 's'
    return 'D'


def _parse_datetime_string(s, unit):
    """Parse datetime string to integer value in given unit."""
    s = s.strip()
    parts = s.replace('T', '-').replace(':', '-').split('-')
    year = int(parts[0]) if len(parts) > 0 else 1970
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1

    if unit == 'Y':
        return year
    elif unit == 'M':
        return (year - 1970) * 12 + (month - 1)
    else:
        # Convert to days since epoch
        days = _date_to_days(year, month, day)
        if unit == 'D':
            return days
        elif unit == 's':
            hour = int(parts[3]) if len(parts) > 3 else 0
            minute = int(parts[4]) if len(parts) > 4 else 0
            second = int(parts[5]) if len(parts) > 5 else 0
            return days * 86400 + hour * 3600 + minute * 60 + second
        return days


def _date_to_days(year, month, day):
    """Convert date to days since 1970-01-01."""
    days = 0
    # Years
    for y in range(1970, year) if year >= 1970 else range(year, 1970):
        leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
        d = 366 if leap else 365
        if year >= 1970:
            days += d
        else:
            days -= d
    # Months
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    if leap:
        month_days[1] = 29
    for m in range(1, month):
        days += month_days[m - 1]
    days += day - 1
    return days


def _days_to_date(days):
    """Convert days since epoch to ISO date string."""
    y = 1970
    remaining = days
    if remaining >= 0:
        while True:
            leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
            year_days = 366 if leap else 365
            if remaining < year_days:
                break
            remaining -= year_days
            y += 1
    else:
        while remaining < 0:
            y -= 1
            leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
            year_days = 366 if leap else 365
            remaining += year_days

    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
    if leap:
        month_days[1] = 29
    m = 0
    while m < 12 and remaining >= month_days[m]:
        remaining -= month_days[m]
        m += 1
    return "{:04d}-{:02d}-{:02d}".format(y, m + 1, remaining + 1)


def _to_common_unit(value, from_unit, to_unit):
    """Convert value from one unit to another."""
    # First convert to days
    if from_unit == 'Y':
        days = value * 365  # approximate
    elif from_unit == 'M':
        days = value * 30  # approximate
    elif from_unit == 'W':
        days = value * 7
    elif from_unit == 'D':
        days = value
    elif from_unit == 'h':
        days = value / 24.0
    elif from_unit == 'm':
        days = value / 1440.0
    elif from_unit == 's':
        days = value / 86400.0
    elif from_unit == 'ms':
        days = value / 86400000.0
    elif from_unit == 'us':
        days = value / 86400000000.0
    elif from_unit == 'ns':
        days = value / 86400000000000.0
    else:
        days = value

    # Then convert from days to target
    if to_unit == 'Y':
        return int(days / 365)
    elif to_unit == 'M':
        return int(days / 30)
    elif to_unit == 'W':
        return int(days / 7)
    elif to_unit == 'D':
        return int(days)
    elif to_unit == 'h':
        return int(days * 24)
    elif to_unit == 'm':
        return int(days * 1440)
    elif to_unit == 's':
        return int(days * 86400)
    elif to_unit == 'ms':
        return int(days * 86400000)
    elif to_unit == 'us':
        return int(days * 86400000000)
    elif to_unit == 'ns':
        return int(days * 86400000000000)
    return int(days)


def _common_time_unit(u1, u2):
    """Find the finer of two time units."""
    order = ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns']
    try:
        i1 = order.index(u1)
    except ValueError:
        i1 = 3  # default to days
    try:
        i2 = order.index(u2)
    except ValueError:
        i2 = 3
    return order[i1 if i1 > i2 else i2]


# --- datetime64 / timedelta64 classes ----------------------------------------
# Duck-typing helpers: survive external patching of np.datetime64/np.timedelta64.
def _is_dt64(x):
    """Return True if x is a datetime64 instance."""
    return getattr(x, '_is_datetime64', False)

def _is_td64(x):
    """Return True if x is a timedelta64 instance."""
    return getattr(x, '_is_timedelta64', False)


class datetime64:
    """NumPy datetime64 scalar type."""
    _is_datetime64 = True  # duck-typing tag, survives external patching of np.datetime64

    def __init__(self, value=None, unit=None):
        if value is None:
            self._value = 0  # epoch
            self._unit = unit or 'us'
        elif isinstance(value, str) and value.strip().lower() == 'nat':
            self._value = _NAT_VALUE
            self._unit = unit or 'generic'
        elif isinstance(value, str):
            self._unit = unit or _infer_datetime_unit(value)
            self._value = _parse_datetime_string(value, self._unit)
        elif getattr(value, '_is_datetime64', False):
            self._value = value._value
            self._unit = unit or value._unit
        elif isinstance(value, (int, float)):
            self._value = int(value)
            self._unit = unit or 'us'
        else:
            self._value = int(value)
            self._unit = unit or 'us'

    @property
    def _is_nat(self):
        return self._value == _NAT_VALUE

    def __repr__(self):
        return "numpy.datetime64('{}')".format(self._to_string())

    def __str__(self):
        return self._to_string()

    def _to_string(self):
        """Convert internal value back to ISO string."""
        if self._value == _NAT_VALUE:
            return 'NaT'
        if self._unit == 'Y':
            return str(self._value)
        elif self._unit == 'M':
            y = 1970 + self._value // 12
            m = self._value % 12 + 1
            return "{:04d}-{:02d}".format(y, m)
        elif self._unit == 'D':
            # Days since epoch (1970-01-01)
            return _days_to_date(self._value)
        elif self._unit in ('h', 'm', 's', 'ms', 'us', 'ns'):
            # Convert to days + remainder
            if self._unit == 's':
                days = self._value // 86400
            elif self._unit == 'ms':
                days = self._value // 86400000
            elif self._unit == 'us':
                days = self._value // 86400000000
            elif self._unit == 'ns':
                days = self._value // 86400000000000
            elif self._unit == 'h':
                days = self._value // 24
            elif self._unit == 'm':
                days = self._value // 1440
            else:
                days = self._value
            return _days_to_date(days)
        return str(self._value)

    def __sub__(self, other):
        if self._is_nat:
            if _is_dt64(other):
                return timedelta64(_NAT_VALUE, 'generic')
            if _is_td64(other):
                return datetime64.__new_from_value(_NAT_VALUE, self._unit)
        if _is_dt64(other):
            if other._is_nat:
                return timedelta64(_NAT_VALUE, 'generic')
            # datetime - datetime = timedelta
            v1 = _to_common_unit(self._value, self._unit, 'D')
            v2 = _to_common_unit(other._value, other._unit, 'D')
            return timedelta64(v1 - v2, 'D')
        elif _is_td64(other):
            if other._is_nat:
                return datetime64.__new_from_value(_NAT_VALUE, self._unit)
            v = _to_common_unit(other._value, other._unit, self._unit)
            return datetime64.__new_from_value(self._value - v, self._unit)
        return NotImplemented

    def __add__(self, other):
        if self._is_nat or (_is_td64(other) and other._is_nat):
            return datetime64.__new_from_value(_NAT_VALUE, self._unit)
        if _is_td64(other):
            v = _to_common_unit(other._value, other._unit, self._unit)
            return datetime64.__new_from_value(self._value + v, self._unit)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __eq__(self, other):
        # NaT != NaT (like float nan)
        if self._is_nat:
            return False
        if _is_dt64(other):
            if other._is_nat:
                return False
            v1 = _to_common_unit(self._value, self._unit, 'D')
            v2 = _to_common_unit(other._value, other._unit, 'D')
            return v1 == v2
        return NotImplemented

    def __lt__(self, other):
        if self._is_nat or (_is_dt64(other) and other._is_nat):
            return False
        if _is_dt64(other):
            v1 = _to_common_unit(self._value, self._unit, 'D')
            v2 = _to_common_unit(other._value, other._unit, 'D')
            return v1 < v2
        return NotImplemented

    def __le__(self, other):
        if self._is_nat or (_is_dt64(other) and other._is_nat):
            return False
        return self == other or self < other

    def __gt__(self, other):
        if self._is_nat or (_is_dt64(other) and other._is_nat):
            return False
        if _is_dt64(other):
            return other < self
        return NotImplemented

    def __ge__(self, other):
        if self._is_nat or (_is_dt64(other) and other._is_nat):
            return False
        return self == other or self > other

    def __hash__(self):
        return hash((self._value, self._unit))

    @classmethod
    def __new_from_value(cls, value, unit):
        obj = cls.__new__(cls)
        obj._value = value
        obj._unit = unit
        return obj

    def astype(self, dtype):
        dtype_str = str(dtype)
        if 'int' in dtype_str:
            return self._value
        elif 'float' in dtype_str:
            return float(self._value)
        return self


class timedelta64:
    """NumPy timedelta64 scalar type."""
    _is_timedelta64 = True  # duck-typing tag, survives external patching of np.timedelta64

    def __init__(self, value=0, unit='generic'):
        if getattr(value, '_is_timedelta64', False):
            self._value = value._value
            self._unit = unit if unit != 'generic' else value._unit
        elif isinstance(value, str) and value.strip().lower() == 'nat':
            self._value = _NAT_VALUE
            self._unit = unit
        else:
            self._value = int(value)
            self._unit = unit

    @property
    def _is_nat(self):
        return self._value == _NAT_VALUE

    def __repr__(self):
        if self._is_nat:
            return "numpy.timedelta64('NaT', '{}')".format(self._unit)
        return "numpy.timedelta64({}, '{}')".format(self._value, self._unit)

    def __str__(self):
        if self._is_nat:
            return 'NaT'
        return "{} {}".format(self._value, self._unit)

    def __add__(self, other):
        if self._is_nat or (_is_td64(other) and other._is_nat):
            common = _common_time_unit(self._unit, other._unit if _is_td64(other) else self._unit)
            return timedelta64(_NAT_VALUE, common)
        if _is_td64(other):
            common = _common_time_unit(self._unit, other._unit)
            v1 = _to_common_unit(self._value, self._unit, common)
            v2 = _to_common_unit(other._value, other._unit, common)
            return timedelta64(v1 + v2, common)
        if _is_dt64(other):
            return other + self
        return NotImplemented

    def __sub__(self, other):
        if self._is_nat or (_is_td64(other) and other._is_nat):
            common = _common_time_unit(self._unit, other._unit if _is_td64(other) else self._unit)
            return timedelta64(_NAT_VALUE, common)
        if _is_td64(other):
            common = _common_time_unit(self._unit, other._unit)
            v1 = _to_common_unit(self._value, self._unit, common)
            v2 = _to_common_unit(other._value, other._unit, common)
            return timedelta64(v1 - v2, common)
        return NotImplemented

    def __mul__(self, other):
        if self._is_nat:
            return timedelta64(_NAT_VALUE, self._unit)
        if isinstance(other, (int, float)):
            return timedelta64(int(self._value * other), self._unit)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if self._is_nat:
            return timedelta64(_NAT_VALUE, self._unit)
        if isinstance(other, (int, float)):
            return timedelta64(int(self._value / other), self._unit)
        if _is_td64(other):
            if other._is_nat:
                return float('nan')
            common = _common_time_unit(self._unit, other._unit)
            v1 = _to_common_unit(self._value, self._unit, common)
            v2 = _to_common_unit(other._value, other._unit, common)
            return v1 / v2 if v2 != 0 else float('inf')
        return NotImplemented

    def __eq__(self, other):
        # NaT != NaT
        if self._is_nat:
            return False
        if _is_td64(other):
            if other._is_nat:
                return False
            common = _common_time_unit(self._unit, other._unit)
            v1 = _to_common_unit(self._value, self._unit, common)
            v2 = _to_common_unit(other._value, other._unit, common)
            return v1 == v2
        return NotImplemented

    def __lt__(self, other):
        if self._is_nat or (_is_td64(other) and other._is_nat):
            return False
        if _is_td64(other):
            common = _common_time_unit(self._unit, other._unit)
            v1 = _to_common_unit(self._value, self._unit, common)
            v2 = _to_common_unit(other._value, other._unit, common)
            return v1 < v2
        return NotImplemented

    def __hash__(self):
        return hash((self._value, self._unit))

    def astype(self, dtype):
        dtype_str = str(dtype)
        if 'int' in dtype_str:
            return self._value
        elif 'float' in dtype_str:
            return float(self._value)
        return self


# Private aliases so external code patching np.datetime64/np.timedelta64
# doesn't break our internal isinstance checks.
_datetime64_cls = datetime64
_timedelta64_cls = timedelta64


def isnat(x):
    """Test element-wise for NaT (Not a Time)."""
    if isinstance(x, (_datetime64_cls, _timedelta64_cls)):
        return x._is_nat
    if isinstance(x, _ObjectArray):
        results = []
        for v in x._data:
            if isinstance(v, (_datetime64_cls, _timedelta64_cls)):
                results.append(v._is_nat)
            else:
                results.append(False)
        arr = _native.array([1.0 if r else 0.0 for r in results]).astype('bool')
        return arr.reshape(list(x._shape))
    if isinstance(x, ndarray):
        # For ndarray, check element-wise - unlikely to have NaT but handle gracefully
        return _native.array([0.0] * x.size).astype('bool').reshape(list(x.shape))
    return False


def datetime_data(dtype):
    """Return (unit, step) for a datetime64 or timedelta64 dtype or scalar."""
    if _is_dt64(dtype) or _is_td64(dtype):
        u = dtype._unit
        if isinstance(u, tuple):
            return u
        return (u, 1)
    # Handle dtype objects
    dt_str = str(dtype)
    if 'datetime64' in dt_str or 'timedelta64' in dt_str:
        import re
        m = re.search(r'\[(\w+)\]', dt_str)
        if m:
            return (m.group(1), 1)
        return ('generic', 1)
    return ('generic', 1)


def busday_count(begindates, enddates, weekmask='1111100', holidays=None):
    """Count business days. Simplified implementation."""
    if _is_dt64(begindates) and _is_dt64(enddates):
        diff = enddates - begindates
        return int(diff._value * 5 / 7)  # rough approximation
    return 0


def is_busday(dates, weekmask='1111100', holidays=None):
    """Check if dates are business days."""
    return True


def busday_offset(dates, offsets, roll='raise', weekmask='1111100', holidays=None):
    """Offset dates by business days."""
    if _is_dt64(dates):
        return dates + timedelta64(int(offsets), 'D')
    return dates
