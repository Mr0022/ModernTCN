#!/usr/bin/env python3
"""
Figures for the ModernTCN OFAT hyperparameter-sensitivity sweep.

Reads sensitivity/ofat_results.csv (written by ofat_sensitivity.py) and writes
into sensitivity/figures/ :

  1. ofat_response_mse.(pdf|png)    response curve per hyperparameter, MSE [ln]
  2. ofat_response_qlike.(pdf|png)  the same, QLIKE on the variance scale
  3. ofat_tornado_<metric>.(pdf|png)  signed swing away from the anchor, ranked
  4. ofat_sensitivity_bar.(pdf|png) each factor's range as a % of the anchor
  5. ofat_summary.csv               per-point mean/std, tidy

    python sensitivity/ofat_plots.py
    python sensitivity/ofat_plots.py --metric qlike

DESIGN NOTES

One measure per axis, never a twin axis. Each response panel draws a single
series -- the metric named in its own y-label -- so no legend box is needed;
the figure caption carries the rest.

The tuned anchor is an ANNOTATION, not a third series, so it is marked in ink
rather than in a hue: a coloured diamond would have to be told apart from the
curve it sits on, and against the QLIKE hue no red passes that test (the two
separate by only about 7 units of normal-vision Delta E, well under the 15 floor --
readers with full colour vision would struggle, before considering CVD). Ink at
full contrast is unambiguous on every panel and reads as the annotation it is.

Points where patch_stride had to be clamped to patch_size are drawn hollow:
those configurations differ from the anchor in two ways rather than one, so the
OFAT reading does not strictly hold there and the figure says so.

The tornado's better/worse pair separates well in normal vision but sits in the
CVD floor band, so colour there is redundant encoding only -- side of zero is
the real signal, and both sides are labelled.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ofat_sensitivity import GRIDS, ORDER, load_anchor, value_key  # noqa: E402

# --- palette ----------------------------------------------------------------
# Validated for the pairs that actually co-occur: series hue against the ink
# annotation, and the tornado's two poles against each other.
BLUE, ORANGE, RED, GREEN = '#2a78d6', '#eb6834', '#e34948', '#008300'
INK, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
METRIC_HUE = {'mse': BLUE, 'mae': BLUE, 'qlike': ORANGE,
              'mse_rv': ORANGE, 'mae_rv': ORANGE}
METRIC_LABEL = {'mse': 'MSE [ln]', 'mae': 'MAE [ln]', 'qlike': 'QLIKE [RV]',
                'mse_rv': 'MSE$_{RV}$', 'mae_rv': 'MAE$_{RV}$'}

PRETTY = {
    'seq_len': 'look-back length  (seq_len)',
    'patch_size': 'patch size',
    'patch_stride': 'patch stride',
    'dim': 'model width  (dims)',
    'ffn_ratio': 'ConvFFN ratio',
    'large_size': 'large kernel size',
    'small_size': 'small kernel size',
    'num_blocks': 'blocks in the stage',
    'dropout': 'dropout',
    'head_dropout': 'head dropout',
    'learning_rate': 'learning rate',
    'batch_size': 'batch size',
    'revin': 'RevIN',
}
LOG_X = {'learning_rate'}      # true numeric log axis; the rest are ordinal

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.labelcolor': INK, 'text.color': INK,
    'figure.dpi': 120, 'savefig.bbox': 'tight',
})


def fmt_val(param, v):
    if param == 'learning_rate':
        return '{:g}'.format(v)
    if param in ('dropout', 'head_dropout'):
        return '{:.2f}'.format(v)
    if param == 'revin':
        return 'off' if v == 0 else 'on'
    return '{:d}'.format(int(v)) if float(v).is_integer() else '{:g}'.format(v)


def save(fig, out):
    for ext in ('pdf', 'png'):
        fig.savefig('{}.{}'.format(out, ext), dpi=200)
    plt.close(fig)
    print('wrote {}.pdf / .png'.format(out))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(results_csv):
    # 'value' is a categorical key (repr of the swept value) and must stay text
    # so it matches value_key(); pandas would otherwise coerce it to float.
    df = pd.read_csv(results_csv, dtype={'value': str})
    df['value'] = df['value'].fillna('')
    if 'note' not in df.columns:
        df['note'] = ''
    df['note'] = df['note'].fillna('')
    for c in METRIC_HUE:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    anchor = df[df['param'] == 'anchor']
    if anchor.empty:
        raise SystemExit('No anchor rows in {} -- run the anchor point first.'
                         .format(results_csv))
    return df, anchor


def series_for(df, anchor_rows, anchor_cfg, param, metric):
    """(values, per-seed arrays, means, stds, clamped flags) sorted by value."""
    grid = sorted(set(GRIDS[param]) | {anchor_cfg[param]})
    vals, seeds, means, stds, clamped = [], [], [], [], []
    for v in grid:
        if v == anchor_cfg[param]:
            sub = anchor_rows
        else:
            sub = df[(df['param'] == param) & (df['value'] == value_key(v))]
        s = sub[metric].dropna().values
        if len(s) == 0:
            continue
        vals.append(v)
        seeds.append(s)
        means.append(float(np.mean(s)))
        stds.append(float(np.std(s, ddof=1)) if len(s) > 1 else 0.0)
        clamped.append(bool(sub['note'].astype(str).str.len().gt(0).any()))
    return vals, seeds, np.array(means), np.array(stds), clamped


# ---------------------------------------------------------------------------
# 1/2 -- response curves
# ---------------------------------------------------------------------------

def fig_response(df, anchor_rows, anchor_cfg, metric, params, out, horizon):
    hue = METRIC_HUE[metric]
    # Never more columns than panels: a two-factor sweep should not render as a
    # quarter-full four-column grid with the caption stretched across the gap.
    ncols = min(4, max(1, len(params)))
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    any_clamped = False
    for i, p in enumerate(params):
        ax = axes.flat[i]
        ax.set_visible(True)
        vals, seeds, means, stds, clamped = series_for(
            df, anchor_rows, anchor_cfg, p, metric)
        if len(vals) < 2:
            ax.set_title(PRETTY.get(p, p) + '  (insufficient data)', color=MUTED)
            continue

        if p in LOG_X:
            xpos = np.array(vals, dtype=float)
            ax.set_xscale('log')
        else:
            xpos = np.arange(len(vals))
            ax.set_xticks(xpos)
            ax.set_xticklabels([fmt_val(p, v) for v in vals])

        ax.fill_between(xpos, means - stds, means + stds, color=hue,
                        alpha=0.16, linewidth=0)
        ax.plot(xpos, means, '-', color=hue, lw=2.0, zorder=3)

        # Filled markers for clean points; hollow where patch_stride was clamped,
        # because those differ from the anchor in two factors, not one.
        for xp, m, cl in zip(xpos, means, clamped):
            any_clamped |= cl
            ax.plot([xp], [m], 'o', ms=7, zorder=4,
                    color='white' if cl else hue,
                    mec=hue, mew=1.8)

        rng = np.random.default_rng(0)
        for xp, s in zip(xpos, seeds):
            jit = 0 if p in LOG_X else (rng.random(len(s)) - 0.5) * 0.10
            ax.plot(np.full(len(s), xp) + jit, s, '.', color=hue,
                    alpha=0.30, ms=4, zorder=2)

        av = anchor_cfg[p]
        if av in vals:
            ai = vals.index(av)
            axp = av if p in LOG_X else xpos[ai]
            ax.axvline(axp, color=AXIS, ls='--', lw=0.9, zorder=1)
            ax.plot([axp], [means[ai]], 'D', color=INK, ms=8,
                    mec='white', mew=1.0, zorder=6)

        ax.set_title(PRETTY.get(p, p))
        ax.set_ylabel(METRIC_LABEL[metric])
        ax.grid(axis='x', visible=False)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    caption = ('line = {}-seed mean, band = ±1 sd, dots = individual seeds, '
               'ink ◆ = tuned anchor'.format(len(seeds[0]) if seeds else 0))
    if any_clamped:
        caption += '\nhollow marker = patch_stride clamped to patch_size, so that point moves two factors'
    fig.suptitle('ModernTCN (h={}) one-factor-at-a-time sensitivity  —  test {}\n{}'
                 .format(horizon, METRIC_LABEL[metric], caption),
                 fontsize=11, y=1.004)
    fig.tight_layout()
    save(fig, out)


# ---------------------------------------------------------------------------
# 3 -- tornado
# ---------------------------------------------------------------------------

def fig_tornado(df, anchor_rows, anchor_cfg, metric, params, out, horizon):
    a_mean = float(anchor_rows[metric].mean())
    rows = []
    for p in params:
        vals, seeds, means, stds, _ = series_for(
            df, anchor_rows, anchor_cfg, p, metric)
        if len(vals) < 2:
            continue
        better = max(0.0, a_mean - means.min())     # best gain available
        worse = max(0.0, means.max() - a_mean)      # worst degradation
        rows.append((p, better, worse, better + worse))
    rows.sort(key=lambda r: r[3])                   # biggest swing on top
    if not rows:
        return

    ys = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.8, 0.5 * len(rows) + 1.6))
    for y, (p, better, worse, _) in zip(ys, rows):
        ax.barh(y, -better, color=GREEN, alpha=0.9, height=0.62)
        ax.barh(y, worse, color=RED, alpha=0.9, height=0.62)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY.get(p, p) for p, *_ in rows])
    ax.set_xlabel('Δ test {} against the tuned anchor'.format(METRIC_LABEL[metric]))
    ax.set_title('ModernTCN (h={}): how far each factor moves {} either side of the anchor'
                 .format(horizon, METRIC_LABEL[metric]))
    # Side of zero is the real encoding; colour repeats it.
    ax.legend(handles=[Patch(color=GREEN, label='← better than anchor'),
                       Patch(color=RED, label='worse than anchor →')],
              frameon=False, loc='lower right')
    ax.grid(axis='y', visible=False)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, out)


# ---------------------------------------------------------------------------
# 4 -- relative sensitivity
# ---------------------------------------------------------------------------

def fig_sensitivity_bar(df, anchor_rows, anchor_cfg, metric, params, out, horizon):
    a_mean = float(anchor_rows[metric].mean())
    rows = []
    for p in params:
        vals, seeds, means, stds, _ = series_for(
            df, anchor_rows, anchor_cfg, p, metric)
        if len(vals) < 2:
            continue
        rows.append((p, (means.max() - means.min()) / a_mean * 100.0))
    rows.sort(key=lambda r: r[1])
    if not rows:
        return

    ys = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.8, 0.5 * len(rows) + 1.3))
    ax.barh(ys, [r[1] for r in rows], color=BLUE, alpha=0.9, height=0.62)
    for y, (_, v) in zip(ys, rows):     # selective direct labels
        ax.text(v, y, '  {:.1f}%'.format(v), va='center', ha='left',
                color=MUTED, fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY.get(p, p) for p, _ in rows])
    ax.set_xlabel('{} range across the grid, as % of the anchor'
                  .format(METRIC_LABEL[metric]))
    ax.set_title('ModernTCN (h={}): which hyperparameters matter, locally'
                 .format(horizon))
    ax.set_xlim(0, max(r[1] for r in rows) * 1.12)
    ax.grid(axis='y', visible=False)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, out)


# ---------------------------------------------------------------------------

def write_summary(df, anchor_rows, anchor_cfg, params, out_csv):
    recs = []
    for p in params:
        for metric in ('mse', 'mae', 'qlike'):
            vals, seeds, means, stds, clamped = series_for(
                df, anchor_rows, anchor_cfg, p, metric)
            for v, s, cl in zip(vals, seeds, clamped):
                recs.append({
                    'param': p, 'value': fmt_val(p, v), 'metric': metric,
                    'n': len(s), 'mean': float(np.mean(s)),
                    'std': float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
                    'is_anchor': v == anchor_cfg[p], 'clamped': cl,
                })
    pd.DataFrame(recs).to_csv(out_csv, index=False)
    print('wrote', out_csv)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--results', default='sensitivity/ofat_results.csv')
    ap.add_argument('--outdir', default='sensitivity/figures')
    ap.add_argument('--metric', default='mse',
                    choices=['mse', 'mae', 'qlike', 'mse_rv', 'mae_rv'],
                    help='metric for the tornado and sensitivity-bar figures')
    ap.add_argument('--anchor_json', default='results_optuna/optuna_h1_best.json',
                    help='the anchor the sweep used; must match ofat_sensitivity.py')
    ap.add_argument('--horizon', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df, anchor_rows = load(args.results)
    anchor_cfg, _ = load_anchor(None if args.anchor_json.lower() == 'none'
                                else args.anchor_json)

    present = [p for p in ORDER if not df[df['param'] == p].empty]
    if not present:
        raise SystemExit('Only the anchor is present -- sweep at least one factor.')
    print('plotting {} factor(s): {}'.format(len(present), ', '.join(present)))

    j = os.path.join
    fig_response(df, anchor_rows, anchor_cfg, 'mse', present,
                 j(args.outdir, 'ofat_response_mse'), args.horizon)
    fig_response(df, anchor_rows, anchor_cfg, 'qlike', present,
                 j(args.outdir, 'ofat_response_qlike'), args.horizon)
    fig_tornado(df, anchor_rows, anchor_cfg, args.metric, present,
                j(args.outdir, 'ofat_tornado_' + args.metric), args.horizon)
    fig_sensitivity_bar(df, anchor_rows, anchor_cfg, args.metric, present,
                        j(args.outdir, 'ofat_sensitivity_bar'), args.horizon)
    write_summary(df, anchor_rows, anchor_cfg, present,
                  j(args.outdir, 'ofat_summary.csv'))
    print('\nAll figures in', args.outdir)


if __name__ == '__main__':
    main()
