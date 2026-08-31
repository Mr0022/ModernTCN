import argparse
import os

import torch
from data_provider.data_factory import is_aggregated
from data_provider.splits import SPLITS, DEFAULT_ASSET
from exp.exp_ModernTCN import Exp_Main
import random
import numpy as np
from utils.str2bool import str2bool

def build_parser():
    """The full command-line interface. Also imported by tune_optuna.py, so the
    search reads exactly these defaults instead of keeping a second copy."""
    parser = argparse.ArgumentParser(description='ModernTCN')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021,
                        help='FIRST random seed; run i of --itr uses random_seed + i, so the default --itr 5 covers 2021..2025')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='ModernTCN',
                        help='model name, options: [ModernTCN]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--asset', type=str, default=DEFAULT_ASSET, choices=sorted(SPLITS),
                        help='which split calendar in data_provider/splits.py to use (--data RV only); \'forex\' = train 2010-01-01..2021-12-31 / val 2022-01-01..2023-12-31 / test 2024-01-01..2025-04-07. HAR-RV_RUN.PY reads the same module and folds val into training, so the test window matches')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')


    # forecasting task
    parser.add_argument('--seq_len', type=int, default=22, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1,
                        help='prediction sequence length; for --data RV this is the HORIZON h '
                             '(1 daily, 5 weekly, 22 monthly) and the model emits ONE number, '
                             'the h-day aggregate, instead of an h-step path')




    #ModernTCN
    # Defaults below are the h=1 column of the Bayesian-optimisation search over
    # data/realized_volatility.csv (best objective 0.17672). patch_stride, batch_size
    # and revin already carried the tuned value. The h=5 and h=22 columns live in
    # scripts/RV.sh; pass them explicitly when running those horizons.
    parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
    parser.add_argument('--ffn_ratio', type=int, default=3, help='ffn_ratio')
    parser.add_argument('--patch_size', type=int, default=32, help='the patch size')
    parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')

    parser.add_argument('--num_blocks', nargs='+',type=int, default=[3], help='num_blocks in each stage; its LENGTH is the number of stages, so large_size and small_size must be at least as long')
    parser.add_argument('--large_size', nargs='+',type=int, default=[51], help='big kernel size')
    parser.add_argument('--small_size', nargs='+',type=int, default=[3], help='small kernel size for structral reparam')
    parser.add_argument('--dims', nargs='+',type=int, default=[32,32,32,32], help='dmodels in each stage; needs 4 entries, one per downsampling layer')
    parser.add_argument('--dw_dims', nargs='+',type=int, default=[32,32,32,32])

    parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
    parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
    parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='use_multi_scale fusion')


    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.2993, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.4155, help='dropout')

    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2,
                        help='how many SEEDS to run; each is trained and tested from scratch, metrics are printed per seed and averaged at the end')
    parser.add_argument('--train_epochs', type=int, default=50,
                        help='maximum training epochs; early stopping on validation loss usually ends a run before this')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping patience: stop after this many epochs with no improvement in validation loss')
    parser.add_argument('--learning_rate', type=float, default=0.0055, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    return parser


def finalize_args(args):
    """Derive the settings that follow from the parsed ones, and resolve the
    device. Separate from parsing so a caller that builds args itself gets the
    same treatment a command line does."""

    # Aggregated-target datasets (data_provider/data_loader.Dataset_RV) collapse the
    # whole forward window into ONE number -- ln of the h-day mean RV -- so --pred_len
    # names the horizon h while the head emits a single step. Everything else keeps
    # forecasting a pred_len-step path, so out_len == pred_len there.
    args.aggregate = is_aggregated(args)
    args.out_len = 1 if args.aggregate else args.pred_len
    if args.aggregate and args.label_len != 0:
        # There is no decoder to prime and no h-step path to start: a label window
        # would only pad batch_y with rows nothing reads.
        print('--data {}: forcing --label_len 0 (aggregated target, no decoder).'
              .format(args.data))
        args.label_len = 0

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    return args


def set_seed(seed):
    """Seed every generator that can move a run. Called once per iteration."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize_seeds(rows, args):
    """
    Print the per-seed table and the average over seeds.

    One seed is one number, not a result: re-running with a different
    initialisation moves these losses around, so the mean is what should be
    reported and the standard deviation says how far to trust it.

    Which columns appear depends on the run. An aggregated RV run
    (--data RV) is scored the way HAR-RV_RUN.PY --log is -- MSE and MAE in
    ln(RV) units, the scale the model is fitted on, plus QLIKE on the
    back-transformed variance, where QLIKE is only defined -- and carries
    MSE_RV / MAE_RV on that same variance scale. Any other run reports the
    standardised-unit losses it was trained on.
    """
    if not rows:
        return
    if getattr(args, 'aggregate', False):
        cols = [('MSE', 'MSE[ln]'), ('MAE', 'MAE[ln]'), ('QLIKE', 'QLIKE[RV]'),
                ('MSE_RV', 'MSE_RV'), ('MAE_RV', 'MAE_RV')]
    else:
        cols = [('MSE', 'MSE'), ('MAE', 'MAE'), ('RSE', 'RSE')]
    cols = [c for c in cols if c[0] in rows[0]]

    seeds = [r['seed'] for r in rows]
    head = '  {:<8}'.format('seed') + ''.join('{:>14}'.format(h) for _, h in cols)
    rule = '-' * len(head)
    title = '  SEED SUMMARY  --  {}  |  {} seed(s): {}'.format(
        args.model_id, len(rows),
        ', '.join(str(s) for s in seeds) if len(seeds) <= 8 else
        '{}..{}'.format(seeds[0], seeds[-1]))
    if getattr(args, 'aggregate', False):
        title += '  |  h={}'.format(args.pred_len)

    print('\n' + '=' * len(head))
    print(title)
    print(rule)
    print(head)
    print(rule)
    for r in rows:
        print('  {:<8}'.format(r['seed'])
              + ''.join('{:>14.6f}'.format(r[k]) for k, _ in cols))
    print(rule)
    for label, fn in (('mean', np.mean), ('std', np.std)):
        print('  {:<8}'.format(label)
              + ''.join('{:>14.6f}'.format(fn([r[k] for r in rows]))
                        for k, _ in cols))
    print('=' * len(head))

    os.makedirs('./results', exist_ok=True)
    out = './results/{}_{}_seed_metrics.csv'.format(args.model_id, args.des)
    keys = [k for k, _ in cols]
    with open(out, 'w') as f:
        f.write(','.join(['model_id', 'horizon', 'seed'] + keys) + '\n')
        for r in rows:
            f.write(','.join([args.model_id, str(args.pred_len), str(r['seed'])]
                             + [str(r[k]) for k in keys]) + '\n')
        for label, fn in (('mean', np.mean), ('std', np.std)):
            f.write(','.join([args.model_id, str(args.pred_len), label]
                             + [str(fn([r[k] for r in rows])) for k in keys]) + '\n')
    print('  -> Saved: {}\n'.format(out))



def main(argv=None):
    args = finalize_args(build_parser().parse_args(argv))
    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        seed_metrics = []
        for ii in range(args.itr):
            # One iteration is one SEED, not a repeat of the same run: seeding
            # here (before the model is built) is what makes the initialisation
            # and the batch order differ, so averaging across iterations
            # measures run-to-run spread instead of re-reporting one draw.
            seed = args.random_seed + ii
            set_seed(seed)

            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_pl{}_dim{}_nb{}_lk{}_sk{}_ffr{}_ps{}_str{}_multi{}_merged{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.dims[0],
                args.num_blocks[0],
                args.large_size[0],
                args.small_size[0],
                args.ffn_ratio,
                args.patch_size,
                args.patch_stride,
                args.use_multi_scale,
                args.small_kernel_merged,
                args.des,
                seed)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {} [seed {}, {}/{}]>>>>>>>>>>>>>>'.format(
                setting, seed, ii + 1, args.itr))
            exp.train(setting)

            print('>>>>>>>testing : {} [seed {}]<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(
                setting, seed))
            m = exp.test(setting)
            m['seed'] = seed
            seed_metrics.append(m)
            if args.aggregate:
                print('  [seed {}]  MSE[ln]:{:.6f}  MAE[ln]:{:.6f}  QLIKE:{:.6f}'
                      .format(seed, m['MSE'], m['MAE'], m['QLIKE']))
            else:
                print('  [seed {}]  MSE:{:.6f}  MAE:{:.6f}'
                      .format(seed, m['MSE'], m['MAE']))

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()

        summarize_seeds(seed_metrics, args)
    else:
        set_seed(args.random_seed)
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
