"""
Split calendars, shared by the deep models and HAR-RV_RUN.PY.

This module is the single source of truth for WHERE the train / validation /
test boundaries fall. Both families import it, so neither can drift away from
the other and start reporting losses on different rows.

    Dataset_RV (data_provider/data_loader.py) -> dl_bounds()
    HAR-RV_RUN.PY                             -> har_bounds()

THE CALENDAR

    train  2010-01-01 .. 2021-12-31
    val    2022-01-01 .. 2023-12-31
    test   2024-01-01 .. 2025-04-07

The deep models fit on `train`, early-stop on `val` and are scored on `test`.
HAR-RV is OLS: it has no hyperparameters to tune, so a held-out validation set
would serve no purpose, and har_bounds() folds those months back into the
estimation sample:

    HAR-RV train  2010-01-01 .. 2023-12-31     [= DL train + val]
    HAR-RV test   2024-01-01 .. 2025-04-07     [= DL test, unchanged]

The TEST window is identical for both, which is the whole point: out-of-sample
losses are then computed on the same rows and can be compared number for
number. Rows after 2025-04-07 are outside every window and are simply not used.

BOUNDS ARE DAY-PRECISE

A bound is a tuple of ints, open on that side when None:

    (year, month)         a whole month -- expanded to its FIRST day as a start
                          bound and its LAST day as an end bound, which is the
                          month granularity HAR-RV_RUN.PY's --train-end and
                          friends parse into.
    (year, month, day)    that exact day, inclusive.

Both forms compare and sort correctly against each other as plain tuples, so a
month-granular override from the command line mixes with a day-granular default
without any conversion at the call site.

THE EMBARGO

Y^(h) at row t averages rows t .. t+h-1, so a row assigned to a split on its own
date alone can carry a target that reaches h-1 rows past the boundary. At the
train|test seam those are training labels built from test data; at the far edge
of the test window they make the test row count depend on where the CSV happens
to stop rather than on the boundary. horizon_month_mask() admits a row only when
its whole target window closes inside the same window, so each window keeps
(its rows) - h + 1 origins.

The deep models get this for free: Dataset_RV slices the split first and then
enumerates len - seq_len - h + 1 windows inside it, so a target can never cross
a border there. The embargo is what puts HAR-RV under the same rule.

pandas and numpy only -- importing this from HAR-RV_RUN.PY must not drag torch
or sklearn into an OLS run.
"""

import calendar

import numpy as np
import pandas as pd

# Inclusive (year, month[, day]) bounds; None is open on that side.
SPLITS = {
    'forex': {
        'train': ((2010, 1, 1), (2021, 12, 31)),
        'val':   ((2022, 1, 1), (2023, 12, 31)),
        'test':  ((2024, 1, 1), (2025, 4, 7)),
    },
    # Kept from the original calendar set; nothing in this repo runs on it
    # today, but --asset crypto stays available for a crypto RV file.
    'crypto': {
        'train': ((2018, 6, 1), (2024, 6, 30)),
        'val':   ((2024, 7, 1), (2025, 6, 30)),
        'test':  ((2025, 7, 1), (2026, 6, 30)),
    },
}

DEFAULT_ASSET = 'forex'


# ------------------------------------------------------------------------------
#  Bound arithmetic
# ------------------------------------------------------------------------------

def _as_timestamp(bound, is_start: bool):
    """
    A bound -> the Timestamp it means, or None when the bound is open.

    A (year, month) bound covers the WHOLE month, so which end of it we want
    depends on which side of the window the bound sits on: the first day for a
    start bound, the last for an end bound. A (year, month, day) bound is that
    day either way.
    """
    if bound is None:
        return None
    if len(bound) == 2:
        year, month = bound
        day = 1 if is_start else calendar.monthrange(year, month)[1]
    elif len(bound) == 3:
        year, month, day = bound
    else:
        raise ValueError('a bound is (year, month) or (year, month, day), '
                         'got {!r}'.format(bound))
    return pd.Timestamp(year=year, month=month, day=day)


def month_of(ts):
    """A timestamp -> the day-precise bound naming it."""
    ts = pd.Timestamp(ts)
    return (ts.year, ts.month, ts.day)


def fmt_month(bound) -> str:
    """A bound -> the string every table, figure title and export prints."""
    if bound is None:
        return 'open'
    if len(bound) == 2:
        return '{:04d}-{:02d}'.format(*bound)
    return '{:04d}-{:02d}-{:02d}'.format(*bound)


def to_ordinal(bound) -> int:
    """
    A bound -> a day ordinal, for ordering two bounds against each other.

    Month bounds map to their first day. That is direction-agnostic on purpose:
    the ordering it induces over (year, month) pairs is the same one a month
    ordinal would give, so an overlap check written for months keeps its meaning
    while day-precise bounds get compared to the day.
    """
    ts = _as_timestamp(bound, is_start=True)
    if ts is None:
        raise ValueError('an open bound has no ordinal')
    return ts.toordinal()


def next_month(bound):
    """
    The bound immediately AFTER an end bound -- where the next window opens.

    A month end rolls to the following month, a day end to the following day,
    so `next_month(train_end)` is the natural default for an unset test start on
    either granularity.
    """
    if bound is None:
        raise ValueError('an open bound has no successor')
    if len(bound) == 2:
        year, month = bound
        return (year + 1, 1) if month == 12 else (year, month + 1)
    return month_of(_as_timestamp(bound, is_start=False) + pd.Timedelta(days=1))


# ------------------------------------------------------------------------------
#  Row selection
# ------------------------------------------------------------------------------

def month_mask(index, lo, hi) -> np.ndarray:
    """
    Boolean mask over `index` for the rows inside [lo, hi], both inclusive.

    Either bound may be None, which leaves that side open. Timestamps are
    compared by calendar day, so an intraday stamp on the closing date is still
    admitted.
    """
    idx = pd.DatetimeIndex(index).normalize()
    mask = np.ones(len(idx), dtype=bool)
    lo_ts = _as_timestamp(lo, is_start=True)
    hi_ts = _as_timestamp(hi, is_start=False)
    if lo_ts is not None:
        mask &= np.asarray(idx >= lo_ts)
    if hi_ts is not None:
        mask &= np.asarray(idx <= hi_ts)
    return mask


def horizon_month_mask(index, h: int, lo, hi) -> np.ndarray:
    """
    month_mask(), restricted to the origins whose h-day target CLOSES inside
    the window.

    The target at row t spans rows t .. t+h-1, so a row is admitted only when
    all h of those rows are themselves inside [lo, hi]. That trims h-1 rows from
    each edge -- exactly the rows whose labels would otherwise be built from the
    neighbouring split's data, and exactly the rows Dataset_RV's window
    enumeration already declines to produce.

    `index` must be the gap-free row grid of the WHOLE sample, not a
    pre-filtered split: the rule counts h-1 rows ahead of each candidate origin.
    """
    base = month_mask(index, lo, hi)
    if h <= 1:
        return base
    n = len(base)
    keep = base.copy()
    for k in range(1, h):
        ahead = np.zeros(n, dtype=bool)
        if n > k:
            ahead[:n - k] = base[k:]
        keep &= ahead
    return keep


def row_range(index, lo, hi):
    """
    Positions of the first and last row of [lo, hi] in `index`, both inclusive.

    Positional rather than a mask because the deep models need to slice a
    look-back window in front of the split, which no mask over the split's own
    rows can express.
    """
    pos = np.flatnonzero(month_mask(index, lo, hi))
    if len(pos) == 0:
        raise ValueError('no rows between {} and {}'
                         .format(fmt_month(lo), fmt_month(hi)))
    return int(pos[0]), int(pos[-1])


# ------------------------------------------------------------------------------
#  Per-family bounds
# ------------------------------------------------------------------------------

def dl_bounds(cal, flag):
    """
    (start, end) for one deep-model split. `flag` is 'train', 'val' or 'test'.

    The three windows are used as written: the deep models fit on train,
    early-stop on val and are scored on test.
    """
    if flag not in cal:
        raise KeyError("flag must be one of {}, got '{}'"
                       .format(sorted(cal), flag))
    return cal[flag]


def har_bounds(cal):
    """
    (train_start, train_end, test_start, test_end) for HAR-RV.

    OLS has no hyperparameters, so the validation months are folded into the
    estimation sample -- training runs from the deep models' train start to
    their VALIDATION end. The test window is passed through untouched, which is
    what keeps the two families scored on identical rows.
    """
    train_start = cal['train'][0]
    train_end = cal['val'][1]
    test_start, test_end = cal['test']
    return train_start, train_end, test_start, test_end
