"""Datetime64 and timedelta64 support."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import _ObjectArray, _flat_arraylike_data

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
    'datetime_as_string',
    'busdaycalendar',
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
        if isinstance(other, ndarray):
            from ._helpers import _make_temporal_array
            dt = str(other.dtype)
            vals = _flat_arraylike_data(other)
            if dt.startswith('timedelta64'):
                out = []
                for v in vals:
                    if _is_td64(v):
                        if v._is_nat:
                            out.append(datetime64.__new_from_value(_NAT_VALUE, self._unit))
                        else:
                            vv = _to_common_unit(v._value, v._unit, self._unit)
                            out.append(datetime64.__new_from_value(self._value + vv, self._unit))
                    else:
                        out.append(datetime64.__new_from_value(self._value + int(v), self._unit))
                return _make_temporal_array(out, 'datetime64[{}]'.format(self._unit)).reshape(other.shape)
            if getattr(other.dtype, 'kind', '') in ('i', 'u', 'b'):
                out = [datetime64.__new_from_value(self._value + int(v), self._unit) for v in vals]
                return _make_temporal_array(out, 'datetime64[{}]'.format(self._unit)).reshape(other.shape)
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
        if isinstance(other, ndarray):
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

    def __le__(self, other):
        if self._is_nat or (_is_td64(other) and other._is_nat):
            return False
        return self == other or self < other

    def __gt__(self, other):
        if self._is_nat or (_is_td64(other) and other._is_nat):
            return False
        if _is_td64(other):
            return other < self
        return NotImplemented

    def __ge__(self, other):
        if self._is_nat or (_is_td64(other) and other._is_nat):
            return False
        return self == other or self > other

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
        results = [
            (v._is_nat if isinstance(v, (_datetime64_cls, _timedelta64_cls)) else False)
            for v in x._data
        ]
        arr = _native.array([1.0 if r else 0.0 for r in results]).astype('bool')
        return arr.reshape(list(x._shape))
    if isinstance(x, ndarray):
        # For ndarray, check element-wise - unlikely to have NaT but handle gracefully
        results = [
            (v._is_nat if isinstance(v, (_datetime64_cls, _timedelta64_cls)) else False)
            for v in _flat_arraylike_data(x)
        ]
        return _native.array([1.0 if r else 0.0 for r in results]).astype('bool').reshape(list(x.shape))
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


_WEEKDAY_NAMES = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')


def _bool_array(values, shape=None):
    arr = _native.array([1.0 if bool(v) else 0.0 for v in values]).astype('bool')
    if shape is not None:
        arr = arr.reshape(list(shape))
    return arr


def _coerce_date_scalar(value):
    if _is_dt64(value):
        return value
    if isinstance(value, str):
        return datetime64(value)
    if value is None:
        return datetime64('NaT')
    return datetime64(value)


def _date_scalar_to_day(value):
    dt = _coerce_date_scalar(value)
    if dt._is_nat:
        return None
    text = str(dt)
    if text == 'NaT':
        return None
    if len(text) == 4:
        return _date_to_days(int(text), 1, 1)
    if len(text) == 7:
        return _date_to_days(int(text[:4]), int(text[5:7]), 1)
    if 'T' in text:
        text = text.split('T', 1)[0]
    if len(text) >= 10:
        return _date_to_days(int(text[:4]), int(text[5:7]), int(text[8:10]))
    return _date_to_days(1970, 1, 1)


def _datetime_from_day(days):
    return datetime64(_days_to_date(int(days)), 'D')


def _weekday_index(days):
    return (int(days) + 3) % 7


def _parse_weekmask(weekmask):
    if isinstance(weekmask, str):
        s = weekmask.strip()
        if len(s) == 7 and all(ch in '01' for ch in s):
            mask = [ch == '1' for ch in s]
        else:
            compact = ''.join(s.split())
            if not compact:
                raise ValueError("weekmask cannot be all zeros")
            if any(ch.islower() for ch in compact):
                raise ValueError("invalid business day weekmask string")
            mask = [False] * 7
            i = 0
            while i < len(compact):
                token = compact[i:i + 3]
                if token not in _WEEKDAY_NAMES:
                    raise ValueError("invalid business day weekmask string")
                mask[_WEEKDAY_NAMES.index(token)] = True
                i += 3
    else:
        values = list(weekmask)
        if len(values) != 7:
            raise ValueError("A business day weekmask array must have length 7")
        mask = [bool(v) for v in values]
    if not any(mask):
        raise ValueError("weekmask cannot be all zeros")
    return tuple(mask)


def _normalize_holidays(holidays, weekmask):
    if holidays is None:
        return tuple()
    if isinstance(holidays, (_datetime64_cls, str)):
        items = [holidays]
    elif isinstance(holidays, ndarray):
        items = _flat_arraylike_data(holidays)
    else:
        items = list(holidays)
    seen = set()
    for item in items:
        day = _date_scalar_to_day(item)
        if day is None:
            continue
        if weekmask[_weekday_index(day)]:
            seen.add(day)
    return tuple(sorted(seen))


def _calendar_parts(weekmask='1111100', holidays=None, busdaycal=None):
    default_mask = _parse_weekmask('1111100')
    default_holidays = tuple()
    if busdaycal is not None:
        if weekmask != '1111100' or holidays is not None:
            raise ValueError("Cannot supply both weekmask/holidays and busdaycal")
        return busdaycal._weekmask, busdaycal._holidays
    mask = _parse_weekmask(weekmask)
    hols = _normalize_holidays(holidays, mask)
    return mask, hols


def _is_business_day_scalar(day, weekmask, holidays):
    if day is None:
        return False
    return weekmask[_weekday_index(day)] and day not in holidays


def _roll_business_day(day, roll, weekmask, holidays):
    if day is None:
        if roll == 'raise':
            raise ValueError("NaT input in busday operation")
        return None
    if _is_business_day_scalar(day, weekmask, holidays):
        return day
    if roll == 'raise':
        raise ValueError("Non-business day date in busday_offset")
    if roll == 'nat':
        return None
    if roll in ('forward', 'following', 'modifiedfollowing'):
        probe = day
        while not _is_business_day_scalar(probe, weekmask, holidays):
            probe += 1
        if roll == 'modifiedfollowing' and _days_to_date(probe)[:7] != _days_to_date(day)[:7]:
            return _roll_business_day(day, 'preceding', weekmask, holidays)
        return probe
    if roll in ('backward', 'preceding', 'modifiedpreceding'):
        probe = day
        while not _is_business_day_scalar(probe, weekmask, holidays):
            probe -= 1
        if roll == 'modifiedpreceding' and _days_to_date(probe)[:7] != _days_to_date(day)[:7]:
            return _roll_business_day(day, 'following', weekmask, holidays)
        return probe
    raise ValueError("Invalid roll parameter")


def _offset_business_day(day, offset, roll, weekmask, holidays):
    day = _roll_business_day(day, roll, weekmask, holidays)
    if day is None:
        return None
    offset = int(offset)
    if offset > 0:
        while offset > 0:
            day += 1
            if _is_business_day_scalar(day, weekmask, holidays):
                offset -= 1
    elif offset < 0:
        while offset < 0:
            day -= 1
            if _is_business_day_scalar(day, weekmask, holidays):
                offset += 1
    return day


def _count_business_days_scalar(begin, end, weekmask, holidays):
    if begin is None or end is None:
        return 0
    if begin == end:
        return 0
    step = 1 if end > begin else -1
    count = 0
    day = begin
    while day != end:
        if _is_business_day_scalar(day, weekmask, holidays):
            count += step
        day += step
    return count


def busday_count(begindates, enddates, weekmask='1111100', holidays=None, busdaycal=None):
    """Count business days between begin and end dates."""
    from ._shape import broadcast_arrays
    from ._creation import asarray
    weekmask, holidays = _calendar_parts(weekmask, holidays, busdaycal)
    begin = asarray(begindates)
    end = asarray(enddates)
    begin_b, end_b = broadcast_arrays(begin, end)
    begin_days = [_date_scalar_to_day(v) for v in _flat_arraylike_data(begin_b)]
    end_days = [_date_scalar_to_day(v) for v in _flat_arraylike_data(end_b)]
    result = [
        _count_business_days_scalar(b, e, weekmask, holidays)
        for b, e in zip(begin_days, end_days)
    ]
    out = _native.array([float(v) for v in result]).astype('int64')
    if begin_b.shape:
        out = out.reshape(list(begin_b.shape))
    return int(result[0]) if not begin_b.shape else out


def is_busday(dates, weekmask='1111100', holidays=None, busdaycal=None):
    """Check if dates are business days."""
    from ._creation import asarray
    weekmask, holidays = _calendar_parts(weekmask, holidays, busdaycal)
    arr = asarray(dates)
    days = [_date_scalar_to_day(v) for v in _flat_arraylike_data(arr)]
    result = [_is_business_day_scalar(day, weekmask, holidays) for day in days]
    return _bool_array(result, arr.shape if arr.shape else None) if arr.shape else result[0]


def busday_offset(dates, offsets, roll='raise', weekmask='1111100', holidays=None, busdaycal=None):
    """Offset dates by business days."""
    from ._helpers import _make_temporal_array
    from ._shape import broadcast_arrays
    from ._creation import asarray
    weekmask, holidays = _calendar_parts(weekmask, holidays, busdaycal)
    date_arr = asarray(dates)
    offset_arr = asarray(offsets)
    date_b, offset_b = broadcast_arrays(date_arr, offset_arr)
    date_days = [_date_scalar_to_day(v) for v in _flat_arraylike_data(date_b)]
    offset_vals = [int(v) for v in _flat_arraylike_data(offset_b)]
    result = []
    for day, off in zip(date_days, offset_vals):
        out_day = _offset_business_day(day, off, roll, weekmask, holidays)
        result.append('NaT' if out_day is None else _days_to_date(out_day))
    out = _make_temporal_array(result, 'datetime64[D]')
    if date_b.shape:
        return out.reshape(date_b.shape)
    return out[0]


def datetime_as_string(arr, unit=None, timezone='naive', casting='same_kind'):
    """Convert datetime64 array to string representation."""
    from ._helpers import _ObjectArray
    if _is_dt64(arr) or (isinstance(arr, _ObjectArray) and arr.size > 0 and _is_dt64(arr._data[0])):
        items = _flat_arraylike_data(arr) if hasattr(arr, 'shape') else [arr]
        result = [str(x) for x in items]
        if len(result) == 1 and not hasattr(arr, 'shape'):
            return result[0]
        return _ObjectArray(result, 'str')
    # For ndarray of datetime64
    if hasattr(arr, 'flatten'):
        results = []
        for val in _flat_arraylike_data(arr):
            try:
                results.append(str(val))
            except Exception:
                results.append('NaT')
        return _ObjectArray(results, 'str')
    return str(arr)


class busdaycalendar:
    """Business day calendar."""
    def __init__(self, weekmask='1111100', holidays=None):
        from ._helpers import _make_temporal_array
        self._weekmask = _parse_weekmask(weekmask)
        self._holidays = _normalize_holidays(holidays, self._weekmask)
        self.weekmask = _bool_array(self._weekmask, (7,))
        self.holidays = _make_temporal_array(
            [_days_to_date(day) for day in self._holidays],
            'datetime64[D]',
        )
