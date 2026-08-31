"""
==============================================================================
Hyperparameter search for ModernTCN on realized volatility

Bayesian optimisation with a Tree-structured Parzen Estimator (Optuna's
TPESampler). TPE models p(x | y < y*) and p(x | y >= y*) as Parzen (kernel
density) estimators over the search space and proposes the point maximising
their ratio, so each trial is chosen from what the previous ones revealed
rather than drawn at random.

    python tune_optuna.py --n_trials 100                 # h=1, the default
    python tune_optuna.py --n_trials 100 --pred_len 5    # another horizon

WHAT IS OPTIMISED, AND ON WHICH ROWS

The objective is validation loss, by default MSE in ln(RV) units -- the scale
the model is fitted on and the one HAR-RV reports. --objective mae or qlike
switch the criterion; QLIKE is evaluated on the back-transformed variance,
where it is defined, with the lognormal Jensen correction applied.

The TEST window is never read. Every trial trains on 2010-01-01..2021-12-31 and
is scored on 2022-01-01..2023-12-31, the same split data_provider/splits.py
hands the deep models; the test months stay untouched until the winning
configuration is run through run.py. A search that scores on test does not
produce an out-of-sample number afterwards, whatever the final run reports.

THE SEARCH SPACE

    seq_len         categorical  {22, 35, 70, 180}
    patch_size      categorical  {4, 8, 16, 32}
    patch_stride    categorical  {2, 4, 8}          constrained <= patch_size
    dim             categorical  {32, 64, 128, 256}
    ffn_ratio       int          1 .. 4             linear
    large_size      categorical  {13, 27, 31, 51}
    small_size      categorical  {3, 5, 7}
    num_blocks      int          1 .. 3             linear
    dropout         float        0.0 .. 0.5         linear
    head_dropout    float        0.0 .. 0.5         linear
    learning_rate   float        1e-5 .. 1e-2       log
    batch_size      categorical  {64, 128, 256, 512}
    revin           categorical  {0, 1}

    event_dim       categorical  {4, 8, 16}         FiLMTCN only

The patch_stride <= patch_size constraint rules out exactly one of the twelve
(patch_size, patch_stride) pairs -- 4 with 8. Rather than resample the stride
from a shrinking list, which would give TPE a differently-shaped distribution
in different trials, the trial is pruned. It costs about 8% of the budget and
keeps every completed trial drawn from one consistent space.

event_dim belongs to FiLMTCN, which is NOT part of this repository: there is no
event/exogenous branch in models/. It is suggested only when --model names a
model that is actually registered, so the search space matches the FiLMTCN
table without silently tuning a parameter nothing reads.

RESUMING

Trials are written to a SQLite study as they finish (--storage). Re-running the
same command continues where it stopped rather than starting over, which is
what makes a long search survive a disconnected Colab runtime.
==============================================================================
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    sys.exit("optuna is not installed. Run:  pip install optuna")

from exp.exp_ModernTCN import Exp_Main
from run import build_parser, finalize_args, set_seed
from utils.metrics import QLIKE


# ==============================================================================
#  SEARCH SPACE
# ==============================================================================

SPACE = {
    'seq_len':      [22, 35, 70, 180],
    'patch_size':   [4, 8, 16, 32],
    'patch_stride': [2, 4, 8],
    'dim':          [32, 64, 128, 256],
    'large_size':   [13, 27, 31, 51],
    'small_size':   [3, 5, 7],
    'batch_size':   [64, 128, 256, 512],
    'revin':        [0, 1],
    'event_dim':    [4, 8, 16],          # FiLMTCN only
}


def suggest(trial, model: str) -> dict:
    """
    Draw one configuration. Named exactly as the search-space table.

    Every distribution is declared unconditionally so TPE sees the same space in
    every trial; the one constraint is enforced afterwards by the caller, not by
    narrowing a distribution here.
    """
    params = {
        'seq_len':       trial.suggest_categorical('seq_len', SPACE['seq_len']),
        'patch_size':    trial.suggest_categorical('patch_size', SPACE['patch_size']),
        'patch_stride':  trial.suggest_categorical('patch_stride', SPACE['patch_stride']),
        'dim':           trial.suggest_categorical('dim', SPACE['dim']),
        'ffn_ratio':     trial.suggest_int('ffn_ratio', 1, 4),
        'large_size':    trial.suggest_categorical('large_size', SPACE['large_size']),
        'small_size':    trial.suggest_categorical('small_size', SPACE['small_size']),
        'num_blocks':    trial.suggest_int('num_blocks', 1, 3),
        'dropout':       trial.suggest_float('dropout', 0.0, 0.5),
        'head_dropout':  trial.suggest_float('head_dropout', 0.0, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size':    trial.suggest_categorical('batch_size', SPACE['batch_size']),
        'revin':         trial.suggest_categorical('revin', SPACE['revin']),
    }
    if model.lower() == 'filmtcn':
        params['event_dim'] = trial.suggest_categorical('event_dim', SPACE['event_dim'])
    return params


# ==============================================================================
#  ONE TRIAL
# ==============================================================================

def build_args(cli, params):
    """A trial's configuration as the Namespace run.py would have produced."""
    argv = [
        '--is_training', '1',
        '--model_id', 'tune_h{}'.format(cli.pred_len),
        '--model', cli.model,
        '--data', cli.data,
        '--root_path', cli.root_path,
        '--data_path', cli.data_path,
        '--features', 'S', '--target', 'RV', '--asset', cli.asset, '--freq', 'd',
        '--enc_in', '1', '--label_len', '0',
        '--pred_len', str(cli.pred_len),
        '--seq_len', str(params['seq_len']),
        '--patch_size', str(params['patch_size']),
        '--patch_stride', str(params['patch_stride']),
        '--ffn_ratio', str(params['ffn_ratio']),
        '--large_size', str(params['large_size']),
        '--small_size', str(params['small_size']),
        '--num_blocks', str(params['num_blocks']),
        '--dims'] + [str(params['dim'])] * 4 + [
        '--dw_dims'] + [str(params['dim'])] * 4 + [
        '--dropout', str(params['dropout']),
        '--head_dropout', str(params['head_dropout']),
        '--learning_rate', str(params['learning_rate']),
        '--batch_size', str(params['batch_size']),
        '--revin', str(params['revin']),
        '--itr', '1',
        '--train_epochs', str(cli.train_epochs),
        '--patience', str(cli.patience),
        '--des', 'Tune',
        '--lradj', 'type3',
        '--use_multi_scale', 'False',
        '--small_kernel_merged', 'False',
        '--num_workers', str(cli.num_workers),
    ]
    if 'event_dim' in params:
        argv += ['--event_dim', str(params['event_dim'])]
    return finalize_args(build_parser().parse_args(argv))


def validation_loss(exp, objective: str) -> float:
    """
    Score the fitted model on the VALIDATION split, in reportable units.

    _forward_log returns ln(RV), so MSE and MAE come out on the scale the model
    is fitted on. QLIKE needs variances and is undefined on logs, so it is taken
    after exp() with the lognormal Jensen correction -- the same correction the
    test-time report applies, estimated here from these very residuals, which is
    legitimate because this split is the one being optimised against.
    """
    vali_data, vali_loader = exp._get_data(flag='val')
    pred_log, true_log = exp._forward_log(vali_data, vali_loader)
    if objective == 'mse':
        return float(np.mean((true_log - pred_log) ** 2))
    if objective == 'mae':
        return float(np.mean(np.abs(true_log - pred_log)))
    if objective == 'qlike':
        resid_var = float(np.var(true_log - pred_log))
        return QLIKE(np.exp(pred_log + resid_var / 2.0), np.exp(true_log))
    raise ValueError('unknown objective: ' + objective)


def make_objective(cli):
    def objective(trial):
        params = suggest(trial, cli.model)

        # The one constraint in the table. Pruned rather than resampled: see the
        # module docstring.
        if params['patch_stride'] > params['patch_size']:
            raise optuna.TrialPruned(
                'patch_stride {} > patch_size {}'.format(
                    params['patch_stride'], params['patch_size']))

        args = build_args(cli, params)
        set_seed(cli.seed)

        setting = 'optuna_h{}_t{}'.format(cli.pred_len, trial.number)
        exp = Exp_Main(args)

        def on_epoch(epoch, vali_loss):
            # vali_loss is in standardised units -- monotone in the ln(RV) loss
            # and free, so it is what the pruner watches. The reported objective
            # is still computed in real units after training.
            trial.report(vali_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    'behind the median at epoch {}'.format(epoch))

        try:
            exp.train(setting, epoch_callback=on_epoch)
            value = validation_loss(exp, cli.objective)
            trial.set_user_attr(
                'n_params', int(sum(q.numel() for q in exp.model.parameters())))
        except optuna.TrialPruned:
            raise
        except RuntimeError as e:
            # A large dim x long seq_len x big batch corner can exhaust memory.
            # That is a fact about the configuration, so prune it and continue
            # rather than losing the whole study.
            if 'out of memory' in str(e).lower():
                print('  OOM on this configuration; pruning the trial.')
                raise optuna.TrialPruned('out of memory')
            raise
        finally:
            del exp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print('  trial {:3d}  val {}[{}] = {:.6f}'.format(
            trial.number, cli.objective.upper(),
            'RV' if cli.objective == 'qlike' else 'ln', value))
        return value

    return objective


# ==============================================================================
#  REPORTING
# ==============================================================================

def report(study, cli):
    """Print the outcome, write it to JSON, and print the run.py command for it."""
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    best = study.best_trial
    unit = 'RV' if cli.objective == 'qlike' else 'ln'

    line = '=' * 72
    print('\n' + line)
    print('  BEST CONFIGURATION  --  h={}  |  {} complete, {} pruned'
          .format(cli.pred_len, len(done), len(pruned)))
    print(line)
    print('  objective: validation {}[{}] = {:.6f}   (trial {})'
          .format(cli.objective.upper(), unit, best.value, best.number))
    print('-' * 72)
    for k, v in best.params.items():
        print('  {:<16} {}'.format(k, v))
    print(line)

    os.makedirs(cli.outdir, exist_ok=True)
    stem = os.path.join(cli.outdir, 'optuna_h{}'.format(cli.pred_len))
    with open(stem + '_best.json', 'w') as f:
        json.dump({'horizon': cli.pred_len, 'model': cli.model,
                   'objective': cli.objective, 'objective_unit': unit,
                   'best_value': best.value, 'best_trial': best.number,
                   'n_complete': len(done), 'n_pruned': len(pruned),
                   'params': best.params}, f, indent=2)
    try:
        study.trials_dataframe().to_csv(stem + '_trials.csv', index=False)
    except Exception as e:                       # pandas is optional for this
        print('  (trial table not written: {})'.format(e))
    print('  -> Saved: {}_best.json, {}_trials.csv'.format(stem, stem))

    p = best.params
    cmd = (
        'python -u run.py --is_training 1 --model_id RV_h{h} --model {model}'
        ' --data {data} --root_path {root} --data_path {dp}'
        ' --features S --target RV --asset {asset} --freq d --enc_in 1'
        ' --seq_len {seq_len} --label_len 0 --pred_len {h}'
        ' --patch_size {patch_size} --patch_stride {patch_stride}'
        ' --ffn_ratio {ffn_ratio} --large_size {large_size}'
        ' --small_size {small_size} --num_blocks {num_blocks}'
        ' --dims {dim} {dim} {dim} {dim} --dw_dims {dim} {dim} {dim} {dim}'
        ' --dropout {dropout:.4f} --head_dropout {head_dropout:.4f}'
        ' --learning_rate {learning_rate:.5f} --batch_size {batch_size}'
        ' --revin {revin} --random_seed 2021 --itr 5'
        ' --train_epochs 50 --patience 10 --des Exp --lradj type3'
        ' --use_multi_scale False --small_kernel_merged False'
    ).format(h=cli.pred_len, model=cli.model, data=cli.data, root=cli.root_path,
             dp=cli.data_path, asset=cli.asset, **p)
    print('\n  Re-run the winner over 5 seeds:\n')
    print('    ' + cmd + '\n')
    with open(stem + '_command.sh', 'w') as f:
        f.write(cmd + '\n')


# ==============================================================================
#  MAIN
# ==============================================================================

def parse_cli(argv=None):
    p = argparse.ArgumentParser(
        prog='tune_optuna.py',
        description='TPE (Bayesian) hyperparameter search for ModernTCN on '
                    'realized volatility. Scored on the validation split; the '
                    'test window is never read.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--n_trials', type=int, default=100,
                   help='how many configurations to try')
    p.add_argument('--pred_len', type=int, default=1,
                   help='forecast HORIZON h to tune (1 daily, 5 weekly, 22 monthly). '
                        'One search per horizon: the best configuration for one is '
                        'not the best for another')
    p.add_argument('--objective', choices=['mse', 'mae', 'qlike'], default='mse',
                   help="validation loss to minimise. 'mse'/'mae' are in ln(RV) "
                        "units, 'qlike' on the back-transformed variance")
    p.add_argument('--model', type=str, default='ModernTCN',
                   help="model to tune; 'FiLMTCN' adds event_dim to the space")
    p.add_argument('--data', type=str, default='RV')
    p.add_argument('--root_path', type=str, default='./data/')
    p.add_argument('--data_path', type=str, default='realized_volatility.csv')
    p.add_argument('--asset', type=str, default='forex',
                   help='split calendar in data_provider/splits.py')
    p.add_argument('--train_epochs', type=int, default=50,
                   help='epoch cap per trial')
    p.add_argument('--patience', type=int, default=10,
                   help='early-stopping patience per trial')
    p.add_argument('--seed', type=int, default=2021,
                   help='seed every trial trains under. One seed keeps the search '
                        'affordable and comparable trial to trial; re-run the '
                        'winner over several seeds afterwards, which is what the '
                        'printed command does')
    p.add_argument('--sampler_seed', type=int, default=42,
                   help='seed for TPE itself, so a search is reproducible')
    p.add_argument('--startup_trials', type=int, default=15,
                   help='random trials before TPE starts modelling. Too few and '
                        'the Parzen estimators are fitted to noise')
    p.add_argument('--warmup_epochs', type=int, default=8,
                   help='epochs a trial is allowed before the pruner may stop it')
    p.add_argument('--no_prune', action='store_true',
                   help='run every trial to completion')
    p.add_argument('--storage', type=str, default='sqlite:///optuna_rv.db',
                   help="study database; 'none' keeps the study in memory only")
    p.add_argument('--study_name', type=str, default=None,
                   help='defaults to <model>_h<pred_len>_<objective>')
    p.add_argument('--outdir', type=str, default='./results_optuna')
    p.add_argument('--num_workers', type=int, default=2)
    return p.parse_args(argv)


def build_study(cli):
    """
    Create or reopen the study. Separated from main() so a caller that wants to
    drive study.optimize() itself -- a notebook mirroring the database somewhere
    after each trial, say -- gets the same sampler, pruner and storage.
    """
    sampler = TPESampler(seed=cli.sampler_seed,
                         n_startup_trials=cli.startup_trials,
                         multivariate=True, group=True)
    pruner = (optuna.pruners.NopPruner() if cli.no_prune else
              MedianPruner(n_startup_trials=cli.startup_trials,
                           n_warmup_steps=cli.warmup_epochs))
    return optuna.create_study(
        direction='minimize', sampler=sampler, pruner=pruner,
        study_name=cli.study_name,
        storage=None if cli.storage.lower() == 'none' else cli.storage,
        load_if_exists=True)


def main(argv=None, callbacks=None):
    """
    Run a search. `callbacks` is passed straight to study.optimize and fires
    after every trial -- what a Colab session uses to copy the database out to
    Drive as it goes, so a dropped runtime costs only the trial in flight.
    """
    cli = parse_cli(argv)
    if cli.study_name is None:
        cli.study_name = '{}_h{}_{}'.format(cli.model, cli.pred_len, cli.objective)

    if cli.model.lower() == 'filmtcn':
        sys.exit(
            "--model FiLMTCN: no FiLMTCN is implemented in models/, so event_dim "
            "would be tuned and then ignored. Add the model (and an --event_dim "
            "argument in run.py) first; the search space here already covers it.")

    print('=' * 72)
    print('  OPTUNA / TPE SEARCH  --  {}  h={}'.format(cli.model, cli.pred_len))
    print('=' * 72)
    print('  trials      : {} (+ any already in the study)'.format(cli.n_trials))
    print('  objective   : minimise validation {} on the VALIDATION split'
          .format(cli.objective.upper()))
    print('  scored on   : validation months only -- the test window is not read')
    print('  sampler     : TPESampler(multivariate=True), {} random startup trials'
          .format(cli.startup_trials))
    print('  pruner      : {}'.format(
        'disabled' if cli.no_prune else
        'MedianPruner after {} warm-up epochs'.format(cli.warmup_epochs)))
    print('  storage     : {}'.format(cli.storage))
    print('=' * 72)

    study = build_study(cli)
    study.optimize(make_objective(cli), n_trials=cli.n_trials,
                   gc_after_trial=True, show_progress_bar=False,
                   callbacks=callbacks)
    report(study, cli)
    return study


if __name__ == '__main__':
    main()
