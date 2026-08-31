#!/usr/bin/env python3
"""
OFAT (one-factor-at-a-time) hyperparameter-sensitivity sweep for ModernTCN, h=1.

Every hyperparameter is anchored at the tuned configuration and ONE is swept at a
time over the grid Optuna searched. Each point is trained with ``run.py --itr 5``
so it carries a 5-seed mean +/- std, and the per-seed test metrics are read back
from the CSV run.py writes. The result is a tidy long-format table that
``ofat_plots.py`` turns into figures.

Local, not global. An OFAT curve says how the loss moves when you step away from
the tuned point in ONE direction. It cannot see interactions -- a pair of
parameters that only matters jointly is invisible here -- so the curves are valid
AROUND the optimum and are labelled as such in the figures. For a global picture,
read the Optuna study in optuna-dashboard, which has the whole search.

THE ANCHOR

Taken from results_optuna/optuna_h1_best.json -- what tune_optuna.py writes -- so
a finished search feeds this sweep with no copying. Falls back to the tuned
values baked into run.py's defaults when no search output is present, and
--anchor_json points at any other file.

THE CONSTRAINT, AND WHY IT IS RECORDED

patch_stride <= patch_size. Sweeping patch_size below the anchor stride would
otherwise ask for an impossible configuration, so the stride is clamped -- but
that point then differs from the anchor in TWO ways, which is exactly what OFAT
is supposed to avoid. Rather than clamp silently, every clamped point carries a
note in the CSV and is drawn hollow in the figures, so a reader can see that the
comparison is not clean there.

METRICS

Read from results/<model_id>_<des>_seed_metrics.csv rather than scraped from
stdout: run.py already writes exactly these numbers, and a CSV cannot drift out
of step with a print format. MSE and MAE are in ln(RV) units, QLIKE is on the
back-transformed variance, and MSE_RV / MAE_RV accompany it on that scale.

Run from the ModernTCN-Long-term-forecasting/ directory:

    python sensitivity/ofat_sensitivity.py                    # full sweep
    python sensitivity/ofat_sensitivity.py --params dropout learning_rate
    python sensitivity/ofat_sensitivity.py --quick            # 2 seeds x 10 epochs
    python sensitivity/ofat_sensitivity.py --dry_run          # print the plan only
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Anchor -- the tuned configuration every sweep pivots around.
# These are run.py's own defaults (the h=1 Bayesian-optimisation result); a
# newer search overrides them through results_optuna/optuna_h1_best.json.
# ---------------------------------------------------------------------------
ANCHOR = {
    'seq_len':       22,
    'patch_size':    32,
    'patch_stride':  8,
    'dim':           32,
    'ffn_ratio':     3,
    'large_size':    51,
    'small_size':    3,
    'num_blocks':    3,
    'dropout':       0.4155,
    'head_dropout':  0.2993,
    'learning_rate': 0.00550,
    'batch_size':    128,
    'revin':         1,
}

DEFAULT_ANCHOR_JSON = 'results_optuna/optuna_h1_best.json'

# ---------------------------------------------------------------------------
# Grids -- the value sets tune_optuna.py searches. The two continuous knobs get
# an evenly spaced grid across their searched range (learning_rate on a log
# grid, the scale it was searched on). The anchor value is always folded in, so
# every curve passes through the tuned point.
# ---------------------------------------------------------------------------
GRIDS = {
    'seq_len':       [22, 35, 70, 180],
    'patch_size':    [4, 8, 16, 32],
    'patch_stride':  [2, 4, 8],
    'dim':           [32, 64, 128, 256],
    'ffn_ratio':     [1, 2, 3, 4],
    'large_size':    [13, 27, 31, 51],
    'small_size':    [3, 5, 7],
    'num_blocks':    [1, 2, 3],
    'dropout':       [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'head_dropout':  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2],
    'batch_size':    [64, 128, 256, 512],
    'revin':         [0, 1],
}

ORDER = list(GRIDS)
METRICS = ['mse', 'mae', 'qlike', 'mse_rv', 'mae_rv']

# Column names in run.py's seed CSV, lowercased to the names used here.
CSV_METRICS = {'MSE': 'mse', 'MAE': 'mae', 'QLIKE': 'qlike',
               'MSE_RV': 'mse_rv', 'MAE_RV': 'mae_rv'}

FIELDS = ['param', 'value', 'seed'] + METRICS + ['itr', 'epochs', 'note']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def value_key(v):
    """Stable string key for a swept value; identical in runner and plotter."""
    return repr(v) if isinstance(v, float) else str(v)


def load_anchor(path):
    """
    Overlay a tuning result onto ANCHOR.

    Only keys ANCHOR already knows are taken, so an unrelated field in the JSON
    (a FiLMTCN event_dim, say) cannot silently become a run.py flag.
    """
    if not path or not os.path.exists(path):
        return dict(ANCHOR), None
    with open(path) as f:
        best = json.load(f)
    params = best.get('params', best)
    anchor = dict(ANCHOR)
    taken = []
    for k in ANCHOR:
        if k in params:
            anchor[k] = params[k]
            taken.append(k)
    return anchor, (path, taken, best.get('best_value'))


def build_cmd(anchor, param, value, itr, epochs, patience, model_id, des, cli):
    """
    The run.py command for one OFAT point: anchor everywhere, `value` at `param`.

    num_blocks, large_size and small_size are passed as SINGLE values, not
    repeated four times: in this implementation len(num_blocks) is the number of
    stages, so four entries would silently build a four-stage network instead of
    changing the swept factor.
    """
    cfg = dict(anchor)
    note = ''
    if param is not None:
        cfg[param] = value
    if cfg['patch_stride'] > cfg['patch_size']:
        note = 'patch_stride clamped {}->{}'.format(cfg['patch_stride'], cfg['patch_size'])
        cfg['patch_stride'] = cfg['patch_size']

    dim = str(cfg['dim'])
    cmd = [
        sys.executable, '-u', 'run.py',
        '--is_training', '1',
        '--model_id', model_id,
        '--model', 'ModernTCN',
        '--data', 'RV',
        '--root_path', cli.root_path,
        '--data_path', cli.data_path,
        '--features', 'S', '--target', 'RV',
        '--asset', cli.asset, '--freq', 'd',
        '--enc_in', '1', '--label_len', '0',
        '--pred_len', str(cli.pred_len),
        '--seq_len', str(cfg['seq_len']),
        '--patch_size', str(cfg['patch_size']),
        '--patch_stride', str(cfg['patch_stride']),
        '--ffn_ratio', str(cfg['ffn_ratio']),
        '--num_blocks', str(cfg['num_blocks']),
        '--large_size', str(cfg['large_size']),
        '--small_size', str(cfg['small_size']),
        '--dims', dim, dim, dim, dim,
        '--dw_dims', dim, dim, dim, dim,
        '--dropout', repr(cfg['dropout']),
        '--head_dropout', repr(cfg['head_dropout']),
        '--learning_rate', repr(cfg['learning_rate']),
        '--batch_size', str(cfg['batch_size']),
        '--revin', str(cfg['revin']),
        '--random_seed', str(cli.random_seed),
        '--itr', str(itr),
        '--train_epochs', str(epochs),
        '--patience', str(patience),
        '--des', des,
        '--lradj', cli.lradj,
        '--use_multi_scale', 'False',
        '--small_kernel_merged', 'False',
        '--num_workers', str(cli.num_workers),
    ]
    return cmd, note


def read_seed_metrics(model_id, des):
    """
    Per-seed rows from the CSV run.py writes, mean/std rows dropped.

    Reading the file rather than the log keeps this in step with run.py by
    construction: the same numbers it summarises are the ones parsed here.
    """
    path = os.path.join('results', '{}_{}_seed_metrics.csv'.format(model_id, des))
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r['seed'] in ('mean', 'std'):
                continue
            row = {'seed': int(r['seed'])}
            for src, dst in CSV_METRICS.items():
                row[dst] = float(r[src]) if r.get(src) not in (None, '') else float('nan')
            rows.append(row)
    return rows


def load_done(csv_path):
    """(param, value) pairs already recorded, so an interrupted sweep resumes."""
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            for r in csv.DictReader(f):
                done.add((r['param'], r['value']))
    return done


def append_rows(csv_path, rows):
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# One point
# ---------------------------------------------------------------------------

def run_point(anchor, param, value, cli, itr, epochs, dry_run):
    tag = 'anchor' if param is None else '{}_{}'.format(param, value_key(value))
    model_id = ('OFAT_' + tag).replace('.', 'p').replace('-', 'm')
    des = 'ofat'
    cmd, note = build_cmd(anchor, param, value, itr, epochs, cli.patience,
                          model_id, des, cli)

    print('\n' + '=' * 72)
    print('[OFAT] {}{}'.format(tag, '   (' + note + ')' if note else ''))
    print(' '.join(cmd))
    print('=' * 72, flush=True)
    if dry_run:
        return []

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cli.timeout)
    except subprocess.TimeoutExpired:
        print('[WARN] {}: timed out after {}s; skipping.'.format(tag, cli.timeout))
        return []
    dt = time.time() - t0

    if proc.returncode != 0:
        print('[WARN] {}: run.py exited {} after {:.0f}s. Last stderr:\n{}'
              .format(tag, proc.returncode, dt, proc.stderr[-800:]), flush=True)
        return []

    metrics = read_seed_metrics(model_id, des)
    if len(metrics) < itr:
        print('[WARN] {}: found {}/{} seed rows; skipping this point.\n{}'
              .format(tag, len(metrics), itr, proc.stdout[-600:]), flush=True)
        return []

    pv = '' if param is None else value_key(value)
    rows = [[param or 'anchor', pv, m['seed']] + [m[k] for k in METRICS]
            + [itr, epochs, note] for m in metrics]
    mean_mse = sum(m['mse'] for m in metrics) / len(metrics)
    print('[OK] {}: {} seeds, mean MSE[ln]={:.5f}  ({:.0f}s)'
          .format(tag, len(metrics), mean_mse, dt), flush=True)
    return rows


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='OFAT sensitivity sweep for ModernTCN on realized volatility.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--params', nargs='+', default=ORDER, choices=ORDER,
                    help='which hyperparameters to sweep')
    ap.add_argument('--itr', type=int, default=5, help='seeds per point')
    ap.add_argument('--train_epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--pred_len', type=int, default=1, help='horizon h')
    ap.add_argument('--anchor_json', default=DEFAULT_ANCHOR_JSON,
                    help="tuning result to anchor on; 'none' uses the built-in "
                         'defaults')
    ap.add_argument('--out', default='sensitivity/ofat_results.csv')
    ap.add_argument('--root_path', default='./data/')
    ap.add_argument('--data_path', default='realized_volatility.csv')
    ap.add_argument('--asset', default='forex')
    ap.add_argument('--lradj', default='type3')
    ap.add_argument('--random_seed', type=int, default=2021)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--timeout', type=int, default=3 * 3600,
                    help='per-configuration subprocess timeout, seconds')
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 2 seeds, 10 epochs')
    ap.add_argument('--dry_run', action='store_true',
                    help='print the plan and the commands, train nothing')
    cli = ap.parse_args()

    itr, epochs = cli.itr, cli.train_epochs
    if cli.quick:
        itr, epochs = 2, 10

    anchor, src = load_anchor(None if cli.anchor_json.lower() == 'none'
                              else cli.anchor_json)
    print('=' * 72)
    print('  OFAT SENSITIVITY  --  ModernTCN  h={}'.format(cli.pred_len))
    print('=' * 72)
    if src:
        path, taken, best = src
        print('  anchor    : {} ({} parameter(s){})'
              .format(path, len(taken),
                      ', best value {:.6f}'.format(best) if best else ''))
    else:
        print("  anchor    : built-in tuned defaults (no {} found)"
              .format(cli.anchor_json))
    for k in ORDER:
        print('              {:<15} {}'.format(k, anchor[k]))
    print('  seeds     : {} per point, from {}'.format(itr, cli.random_seed))
    print('  epochs    : {} (early stopping patience {})'.format(epochs, cli.patience))
    print('=' * 72)

    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)
    done = load_done(cli.out)

    plan = []
    if ('anchor', '') not in done:
        plan.append((None, None))
    for p in cli.params:
        for v in sorted(set(GRIDS[p]) | {anchor[p]}):
            if v == anchor[p]:
                continue          # the centre point IS the anchor, trained once
            if (p, value_key(v)) in done:
                continue
            plan.append((p, v))

    print('\nPlan: {} configuration(s) to train, {} already recorded.'
          .format(len(plan), len(done)))
    for p, v in plan:
        print('   - {}'.format('anchor' if p is None else '{} = {}'.format(p, v)))
    print('\nEach is {} seed(s), so {} training run(s) in total.'
          .format(itr, len(plan) * itr))

    if cli.dry_run:
        for p, v in plan:
            run_point(anchor, p, v, cli, itr, epochs, dry_run=True)
        print('\n[dry run] nothing trained.')
        return

    for i, (p, v) in enumerate(plan, 1):
        print('\n########## [{}/{}] ##########'.format(i, len(plan)))
        rows = run_point(anchor, p, v, cli, itr, epochs, dry_run=False)
        if rows:
            append_rows(cli.out, rows)

    print('\nDone. Results in {}'.format(cli.out))
    print('Next:  python sensitivity/ofat_plots.py')


if __name__ == '__main__':
    main()
