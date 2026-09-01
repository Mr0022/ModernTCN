"""
Daily macro news-event features for the aggregated realized-volatility runs.

Turns the raw EUR/USD economic calendar -- one row per scheduled event, with a
time stamp, a currency, an impact rating and a name -- into a per-DAY numeric
matrix aligned to the trading days of the RV series.  Dataset_RV_Events feeds
the look-back slice of that matrix to the event stem and the HORIZON slice to
the FiLM generator (data_provider/data_loader.py).

WHAT THE FEATURES ARE, AND WHY THEY ARE SAFE

The calendar carries no actual / forecast / previous values -- only the fact
that an event is SCHEDULED, plus a vendor impact rating.  So every column here
is a release-calendar variable, known at the forecast origin, never an outcome.
That is what makes the horizon slice a legitimate future covariate rather than
a leak: the model learns "an FOMC statement lands on day 3 of this window", not
what the statement said.

Two blocks of columns:

    n_events, n_events_high/medium/low, n_events_eur/usd
        per-day counts.  Standardised on TRAINING rows only, by the caller.

    evt_<CUR>_<key>
        per-day counts for one event key in one currency.  What a "key" is
        depends on --event_vocab:

          'role' (default)  a normalised ROLE: every 'FOMC Member <person>
                            Speaks' collapses to one cb_member_speech column,
                            Draghi / Lagarde / Trichet to one cb_chief_speech
                            column, all CPI prints to one cpi column, and so
                            on.  ~16 live columns.

          'name'            one column per raw event name, ~100 of them, which
                            is what FilmTCN's data/events.csv holds.  Kept for
                            an exact-granularity ablation.

WHY 'role' IS THE DEFAULT

Officials churn and the raw names churn with them.  Over this calendar, 12 event
names never occur in the training window (2010-2021) yet fire 213 times in test
(23.5% of test events) -- Barr, Collins, Cook, Goolsbee, Hammack, Jefferson,
Kugler, Logan, Musalem, Schmid, Nagel, Cleveland Fed Inflation Expectations.  A
column that is identically zero in training receives no gradient at all (the
gradient into an input weight is proportional to that input), so its embedding
column keeps its random initialisation and then injects a fixed random vector
into the conditioning at test time.  The reverse is just as wasteful: Draghi
(203 training rows), Dudley (152) and Weidmann (132) each own a well-trained
column that never fires again after 2021.

Normalising names to roles maps every one of those dead names onto a column with
hundreds of training examples, and stops retired officials from consuming
capacity.  Whatever the vocabulary, drop_dead_columns() then removes anything
still identically zero over the training rows, so no column can reach test time
untrained.

TIMING ASSUMPTION

An event is assigned to the calendar day of its time stamp, with no session
shift.  37% of the events stamp at 19:00-20:00, so if the realized-variance day
is not cut at midnight in the same zone as these stamps, some evening events
belong to the next RV day instead.  Fixing that needs the RV series' session
convention, which this file does not know; it is the first thing to revisit if
the h=1 event effect looks weaker than expected.
"""

import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Role normalisation
#
# First match wins, so the specific central-bank rules sit in front of the
# bare 'CPI' / 'PPI' substring rules.  Anything unmatched lands in 'other' and
# is reported by build_event_frame, because a silently mis-bucketed event is
# the failure mode worth being loud about.
# ---------------------------------------------------------------------------

_ROLE_RULES = (
    (r'^FOMC (Statement|Press Conference|Economic Projections)$', 'cb_decision'),
    (r'^Fed Announcement$',                                       'cb_decision'),
    (r'^ECB Press Conference$',                                   'cb_decision'),

    (r'^FOMC Meeting Minutes$',                                   'cb_minutes'),
    (r'^ECB Monetary Policy Meeting Accounts$',                   'cb_minutes'),

    (r'^Fed Chair(man)? .+ (Speaks|Testifies)$',                  'cb_chief_speech'),
    (r'^ECB President .+ Speaks$',                                'cb_chief_speech'),

    (r'^FOMC Member .+ Speaks$',                                  'cb_member_speech'),
    (r'^German Buba President .+ Speaks$',                        'cb_member_speech'),

    (r'^(FOMC Financial Stability Report|Fed Monetary Policy Report'
     r'|Fed Chairman Nomination Vote|Fed Gov Nomination Hearings)$', 'cb_report'),
    (r'^ECB (Economic Bulletin|Monthly Bulletin|Financial Stability Review'
     r'|Stress Test Results)$',                                    'cb_report'),

    (r'CPI',                                                      'cpi'),
    (r'PPI',                                                      'ppi'),

    (r'^(Philly Fed Manufacturing Index'
     r'|Cleveland Fed Inflation Expectations)$',                   'survey'),

    (r'^President .+ Speaks$',                                    'govt_speech'),
    (r'^Treasury Sec .+ Speaks$',                                 'govt_speech'),
)

_ROLE_RULES = tuple((re.compile(pat), fam) for pat, fam in _ROLE_RULES)

IMPACT_LEVELS = ('high', 'medium', 'low')


def event_role(name):
    """Normalised role for one raw event name; 'other' when no rule matches."""
    for pattern, family in _ROLE_RULES:
        if pattern.search(name):
            return family
    return 'other'


def _slug(text):
    """Event name -> column-safe key, matching FilmTCN's evt_* naming."""
    return re.sub(r'[^0-9A-Za-z]+', '_', text).strip('_')


# ---------------------------------------------------------------------------
# Calendar -> per-day feature frame
# ---------------------------------------------------------------------------

def load_calendar(path):
    """
    Read the raw calendar CSV into tidy rows: day, currency, impact, event.

    Expects the columns the vendor file carries -- DateTime, Currency, Impact,
    Event -- plus whatever empty trailing columns the export left behind.
    """
    df = pd.read_csv(path)
    df = df[[c for c in df.columns if not str(c).startswith('Unnamed')]]

    missing = {'DateTime', 'Currency', 'Impact', 'Event'} - set(df.columns)
    if missing:
        raise ValueError('{}: calendar is missing column(s) {}'
                         .format(os.path.basename(path), sorted(missing)))

    ts = pd.to_datetime(df['DateTime'], format='%m/%d/%Y %H:%M', errors='coerce')
    if ts.isna().any():
        # fall back to a free parse for the rows the strict format rejected
        loose = pd.to_datetime(df.loc[ts.isna(), 'DateTime'], errors='coerce')
        ts = ts.fillna(loose)
    n_bad = int(ts.isna().sum())
    if n_bad:
        print('  WARNING: dropped {} calendar row(s) with an unparsable DateTime.'
              .format(n_bad))

    out = pd.DataFrame({
        'day': ts.dt.normalize(),
        'currency': df['Currency'].astype(str).str.strip().str.upper(),
        # 'High Impact Expected' -> 'high'
        'impact': df['Impact'].astype(str).str.strip().str.split().str[0].str.lower(),
        'event': df['Event'].astype(str).str.strip(),
    })
    return out.loc[ts.notna()].reset_index(drop=True)


def build_event_frame(calendar_path, vocab='role', verbose=True):
    """
    Per-day event features, indexed by calendar day.

    Returns (frame, count_cols).  `frame` holds the count block followed by the
    evt_* block; `count_cols` names the columns the caller must standardise.
    Days with no scheduled event simply do not appear -- align_to_index() fills
    them with zeros, which is what "nothing on the calendar" means.
    """
    if vocab not in ('role', 'name'):
        raise ValueError("event_vocab must be 'role' or 'name', got %r" % (vocab,))

    ev = load_calendar(calendar_path)
    ev['key'] = (ev['event'].map(event_role) if vocab == 'role'
                 else ev['event'].map(_slug))

    if vocab == 'role':
        unmapped = sorted(ev.loc[ev['key'] == 'other', 'event'].unique())
        if unmapped and verbose:
            print('  WARNING: {} event name(s) fell through the role rules into '
                  "'other': {}".format(len(unmapped), unmapped[:8]))

    days = pd.Index(sorted(ev['day'].unique()), name='date')

    counts = pd.DataFrame(index=days)
    counts['n_events'] = ev.groupby('day').size()
    for level in IMPACT_LEVELS:
        counts['n_events_' + level] = (ev[ev['impact'] == level]
                                       .groupby('day').size())
    for cur in ('EUR', 'USD'):
        counts['n_events_' + cur.lower()] = (ev[ev['currency'] == cur]
                                             .groupby('day').size())
    counts = counts.fillna(0.0)

    # one column per (currency, key), holding that day's count for the pair
    wide = (ev.assign(col='evt_' + ev['currency'] + '_' + ev['key'])
              .groupby(['day', 'col']).size()
              .unstack(fill_value=0)
              .reindex(days, fill_value=0)
              .sort_index(axis=1))

    frame = pd.concat([counts, wide], axis=1).astype(np.float32)
    count_cols = list(counts.columns)
    if verbose:
        print('  calendar: {} rows, {} day(s) with events, vocab={!r} -> {} '
              'event column(s)'.format(len(ev), len(days), vocab, wide.shape[1]))
    return frame, count_cols


def align_to_index(frame, index):
    """
    Reindex a per-day event frame onto the trading days of the RV series.

    Trading days absent from the calendar become all-zero rows: nothing was
    scheduled.  Calendar days that are not trading days are dropped -- their
    events fall outside every window the model is ever asked about.
    """
    aligned = frame.reindex(pd.DatetimeIndex(index).normalize()).fillna(0.0)
    return aligned.astype(np.float32)


def drop_dead_columns(values, columns, train_slice, verbose=True):
    """
    Remove event columns that are identically zero over the TRAINING rows.

    Such a column can never receive gradient -- the gradient into an input's
    weight is proportional to that input -- so its embedding stays at its random
    initialisation and injects a fixed random vector whenever the column fires
    out of sample.  Dropping it is strictly better than embedding noise.

    `train_slice` is the (start, stop) half-open row range of the training
    window.  Count columns are exempt: they are dense by construction.
    """
    lo, hi = train_slice
    keep, dropped = [], []
    for j, name in enumerate(columns):
        if name.startswith('evt_') and not np.any(values[lo:hi, j]):
            dropped.append(name)
        else:
            keep.append(j)
    if dropped and verbose:
        print('  dropped {} event column(s) never active in training: {}{}'
              .format(len(dropped), dropped[:6],
                      ' ...' if len(dropped) > 6 else ''))
    return values[:, keep], [columns[j] for j in keep]


def _main():
    """Dump the aligned feature matrix to a CSV for inspection or for HAR-X."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--calendar', required=True, help='raw calendar CSV')
    ap.add_argument('--vocab', default='role', choices=['role', 'name'])
    ap.add_argument('--out', default='events.csv')
    ap.add_argument('--rv', default=None,
                    help='optional RV csv; when given, rows are aligned to its '
                         "'date' column instead of to the calendar's own days")
    args = ap.parse_args()

    frame, _ = build_event_frame(args.calendar, vocab=args.vocab)
    if args.rv:
        rv = pd.read_csv(args.rv)
        frame = align_to_index(frame, pd.to_datetime(rv['date']))
    frame.index.name = 'date'
    frame.to_csv(args.out)
    print('wrote {} ({} rows x {} feature columns)'
          .format(args.out, len(frame), frame.shape[1]))


if __name__ == '__main__':
    _main()
