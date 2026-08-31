from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import ModernTCN
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        # Length of the emitted forecast. run.py sets it to 1 for aggregated
        # targets (--data RV: one number, the ln of the h-day mean RV) and to
        # pred_len otherwise; the fallback keeps other entry points working.
        if not hasattr(args, 'out_len'):
            args.out_len = args.pred_len
        if not hasattr(args, 'aggregate'):
            args.aggregate = False
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'ModernTCN':ModernTCN,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.out_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                            # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark)
                        # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.out_len:, f_dim:]
                batch_y = batch_y[:, -self.args.out_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.out_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                            #outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.out_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.out_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark)
                        # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.out_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.out_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if self.args.call_structural_reparam and hasattr(self.model, 'structural_reparam'):
            self.model.structural_reparam()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.out_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                            # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark)
                        # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.out_len:, f_dim:]
                batch_y = batch_y[:, -self.args.out_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        # Concatenated, not stacked: the aggregated RV loaders keep the last
        # partial batch (every test window has to be scored), so the batches are
        # not all the same length and np.array() could not build one array.
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

        # The losses above are in the standardised units the network is trained
        # in. For an aggregated RV run, restate them in ln(RV) and in RV so they
        # can be read next to HAR-RV_RUN.PY --log.
        if getattr(self.args, 'aggregate', False):
            self._report_rv(setting, folder_path, test_data, preds, trues)
        return

    # ==========================================================================
    #  Aggregated realized-volatility evaluation  (--data RV)
    #
    #  The network is fitted on standardised ln(RV) and predicts ONE number per
    #  window: ln of the mean RV over the next h days. The MSE printed by test()
    #  is therefore in standardised units, which nothing else is measured in. The
    #  block below re-states the same forecasts in the two scales HAR-RV_RUN.PY
    #  --log reports, so the two families can be put in one table:
    #
    #    ln(RV) units : MSE / MAE, the scale the model is actually fitted on.
    #    RV units     : QLIKE, MSE_RV, MAE_RV after exp(), which is defined only
    #                   for variances.
    #
    #  Because the target is ln(ARITHMETIC forward mean) -- log outside the sum,
    #  exactly as in HAR-RV_RUN.PY -- exp() of it recovers the raw h-day average
    #  itself, so these numbers sit beside a HAR-RV run at EVERY horizon, not
    #  only at h=1.
    # ==========================================================================

    def _forward_log(self, data_set, loader):
        """
        One inference pass, returned in ln(RV) units.

        The dataset standardised both channels with the SAME training mean and
        standard deviation, so undoing it is one affine map and the pair comes
        back on the scale HAR-RV is scored on.
        """
        preds, trues = [], []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.out_len:, f_dim:]
                batch_y = batch_y[:, -self.args.out_len:, f_dim:]
                preds.append(outputs.detach().cpu().numpy().reshape(-1))
                trues.append(batch_y.detach().cpu().numpy().reshape(-1))
        if was_training:
            self.model.train()
        pred = data_set.inverse_transform(np.concatenate(preds))
        true = data_set.inverse_transform(np.concatenate(trues))
        return pred, true

    def _jensen_resid_var(self):
        """
        Residual variance of the log forecast, for the lognormal back-transform

            E[RV | F] = exp( E[ln RV | F] + sigma^2 / 2 ).

        exp() alone returns the conditional MEDIAN, which understates the mean by
        exp(sigma^2/2) -- a systematic bias that does not average out.

        HAR-RV takes sigma^2 from its TRAINING residuals; for a network those are
        optimistically small, so the honest analogue is the VALIDATION residuals,
        which the model was early-stopped on but never fitted to. Returns 0.0 if
        the validation split is unusable, which just leaves exp() uncorrected.
        """
        try:
            vali_data, vali_loader = self._get_data(flag='val')
            pred, true = self._forward_log(vali_data, vali_loader)
        except Exception as e:
            print('  Jensen correction skipped ({}); reporting exp() alone.'.format(e))
            return 0.0
        return float(np.var(true - pred))

    def _report_rv(self, setting, folder_path, test_data, preds, trues):
        """Print and export the HAR-comparable losses for one aggregated run."""
        h = self.args.pred_len
        pred_log = test_data.inverse_transform(np.asarray(preds).reshape(-1))
        true_log = test_data.inverse_transform(np.asarray(trues).reshape(-1))

        resid_var = self._jensen_resid_var()
        true_rv = np.exp(true_log)
        pred_rv = np.exp(pred_log + resid_var / 2.0)     # Jensen-corrected
        pred_rv_naive = np.exp(pred_log)                 # median, uncorrected

        def _qlike(actual, predicted):
            # QLIKE (Patton, 2011): mean( RV/RV_hat - ln(RV/RV_hat) - 1 ).
            # exp() cannot return a non-positive variance, so the floor HAR-RV
            # needs for its levels fit never binds here; it is kept only so a
            # numerical underflow to 0.0 cannot produce an infinity.
            floor = 1e-4 * float(np.mean(actual))
            safe = np.where(predicted <= 0, floor, predicted)
            ratio = actual / safe
            return float(np.mean(ratio - np.log(ratio) - 1)), int((predicted <= 0).sum())

        qlike, n_neg = _qlike(true_rv, pred_rv)
        qlike_naive, _ = _qlike(true_rv, pred_rv_naive)
        m = {
            'model': self.args.model,
            'horizon': h,
            'split': 'test',
            'scale': 'ln_RV',
            'MSE': float(np.mean((true_log - pred_log) ** 2)),   # ln(RV) units
            'MAE': float(np.mean(np.abs(true_log - pred_log))),  # ln(RV) units
            'QLIKE': qlike,
            'QLIKE_naive': qlike_naive,
            'MSE_RV': float(np.mean((true_rv - pred_rv) ** 2)),
            'MAE_RV': float(np.mean(np.abs(true_rv - pred_rv))),
            'n_neg_pred': n_neg,
            'pct_neg': 100.0 * n_neg / len(pred_rv),
            'n_obs': len(pred_rv),
            'resid_var_val': resid_var,
        }

        line = ('h={horizon}  n={n_obs}  MSE[ln]:{MSE:.6f}  MAE[ln]:{MAE:.6f}  '
                'QLIKE:{QLIKE:.6f}  MSE_RV:{MSE_RV:.8f}  MAE_RV:{MAE_RV:.8f}'
                .format(**m))
        print('-' * 72)
        print('AGGREGATED RV FORECAST  --  target = ln( mean RV over next '
              '{} day(s) )'.format(h))
        print('  ' + line)
        print('  Jensen correction used sigma^2 = {:.6f} (validation log '
              'residuals); QLIKE without it: {:.6f}'.format(resid_var, qlike_naive))
        print('  MSE/MAE marked [ln] are in ln(RV) units and are NOT comparable')
        print('  to raw-RV losses. QLIKE / MSE_RV / MAE_RV are on the')
        print('  back-transformed variance scale and ARE comparable to HAR-RV.')
        print('-' * 72)

        with open('result_rv.txt', 'a') as f:
            f.write(setting + '  \n')
            f.write(line + '\n\n')

        # Same column names HAR-RV_RUN.PY writes to har_rv_log_all_metrics.csv,
        # so the two files concatenate without renaming anything.
        cols = ['model', 'horizon', 'split', 'scale', 'MSE', 'MAE', 'QLIKE',
                'QLIKE_naive', 'MSE_RV', 'MAE_RV', 'n_neg_pred', 'pct_neg',
                'n_obs', 'resid_var_val']
        with open(os.path.join(folder_path, 'rv_metrics.csv'), 'w') as f:
            f.write(','.join(cols) + '\n')
            f.write(','.join(str(m[c]) for c in cols) + '\n')

        # Per-window forecasts, dated by the first day of the target window, in
        # the same shape as har_rv_log_h*_fitted.csv.
        dates = test_data.target_dates()[:len(pred_log)]
        with open(os.path.join(folder_path, 'rv_forecasts.csv'), 'w') as f:
            f.write('date,Y_h,fitted,residual,actual_RV,fitted_RV\n')
            for d, a, p, ar, pr in zip(dates, true_log, pred_log, true_rv, pred_rv):
                f.write('{},{},{},{},{},{}\n'.format(
                    str(d.date()), a, p, a - p, ar, pr))
        print('  -> Saved: {}rv_metrics.csv, {}rv_forecasts.csv'
              .format(folder_path, folder_path))
        return m

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.out_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                            # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark)
                        # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
