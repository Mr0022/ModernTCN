import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.splits import SPLITS, DEFAULT_ASSET, dl_bounds, row_range, fmt_month
from data_provider.calendar_events import (build_event_frame, align_to_index,
                                           drop_dead_columns)
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_RV(Dataset):
    """
    Realized volatility, HAR-RV style: AGGREGATED horizon target on the LOG scale.

    This loader differs from Dataset_Custom in exactly the two ways HAR-RV_RUN.PY
    --log does, and it does BOTH unconditionally -- there is no raw-RV mode and no
    step-by-step mode here.

    1. AGGREGATED TARGET.  One number per window -- the h-day AVERAGE, not an
       h-step path:

           Y_t^(h) = (1/h) * Sum_{k=1}^{h} RV_{t+k}

       so --pred_len is the HORIZON h (1 daily, 5 weekly, 22 monthly) while the
       head emits a single step (run.py sets args.out_len = 1).  At h=5 the model
       predicts the mean of the next five days, once; it is never asked for the
       five individual days.

    2. LOG SCALE, LOG OF MEAN.  The input channel is ln(RV) and the target is the
       log of the ARITHMETIC forward mean -- the log sits OUTSIDE the sum, the
       same convention HAR-RV_RUN.PY uses for its regressors and its target:

           x_t     =     ln( RV_t )
           Y_t^(h) = ln( (1/h) * Sum_{k=1}^{h} RV_{t+k} )

       Log-of-mean, NOT mean-of-logs: (1/h) * Sum ln RV_{t+k} is the log of a
       GEOMETRIC forward mean, a systematically smaller and much smoother object
       that a single spike lifts only through an h-th root.  With log-of-mean,
       exp(Y_t^(h)) is precisely the raw h-day average, so a back-transformed
       forecast is scored against exactly what HAR-RV predicts -- at every
       horizon, not just at h=1 where the two means coincide.

       Sum vs mean is cosmetic on this scale, since ln(sum) = ln(mean) + ln(h)
       and the constant is absorbed by the model; the mean is what keeps losses
       on one scale across horizons, and is Corsi's (2009) convention.

       At h=1 the whole construction collapses to plain next-day ln(RV).

    Windows.  The input covers rows [t-seq_len+1 .. t] and the target covers rows
    [t+1 .. t+h], so the information set ends strictly one row before the target
    window opens -- a genuine forecast, no look-ahead.  Enumeration is
    len(split) - seq_len - h + 1, which also embargoes the split seams: no
    training window's target can reach into validation rows, and no validation
    window's into test.  That is the same rule horizon_month_mask applies on the
    HAR-RV side, so both families forecast the same rows.

    Splits.  Calendar windows from data_provider/splits.py, the same module
    HAR-RV_RUN.PY reads -- train 2010-01-01..2021-12-31, validation
    2022-01-01..2023-12-31, test 2024-01-01..2025-04-07 on the default calendar.
    The forecast origins of a split are exactly its own rows; the seq_len rows
    in front of it are look-back inputs only, the way HAR's 22-day rolling
    window reaches back before its first estimation row.  HAR folds validation
    into training (OLS has nothing to tune) and keeps the same test window, so
    both families are scored on identical rows.  Rows outside every window --
    here, after 2025-04-07 -- are not used at all.  The one place the two
    families cannot match is the START of training: HAR loses 22 rows to its
    monthly component, this loader loses seq_len rows, because neither can look
    back past the first row in the file.

    Standardisation.  StandardScaler is fitted on the TRAINING ln(RV) rows only.
    The target is built on the raw variance scale first, logged, and only then
    standardised with those same training statistics -- the log is non-linear, so
    aggregating a standardised series would not produce the same object.  Input
    and target therefore live in identical units, which is what lets RevIN's
    per-window de-normalisation apply to the aggregated output unchanged.

    Non-positive RV rows are dropped -- exchange holidays with no trading rather
    than genuine zero-variance days.  ln(RV) is undefined there and a zero actual
    makes QLIKE undefined.  HAR-RV_RUN.PY drops the same rows, so the two
    families see the same calendar.

    Columns.  A 'date' column plus one of 'RV' (raw variance) or 'ln_RV' (already
    logged; exponentiated once so the forward windows can be averaged on the
    variance scale).  --target names the column explicitly when the file carries
    several.  Time stamps are built for interface compatibility only: ModernTCN's
    forward_feature ignores its `te` argument, so nothing downstream reads them.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='realized_volatility.csv',
                 target='RV', scale=True, timeenc=0, freq='d', asset=None):
        # size [seq_len, label_len, pred_len]; pred_len is the HORIZON h here
        if size is None:
            self.seq_len = 96
            self.label_len = 0
            self.horizon = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.horizon = size[2]
        assert self.horizon >= 1, 'pred_len (= horizon h) must be >= 1'

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.asset = DEFAULT_ASSET if asset is None else asset
        if self.asset not in SPLITS:
            raise KeyError("unknown --asset '{}'; data_provider/splits.py "
                           'defines {}'.format(self.asset, sorted(SPLITS)))

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_rv__(self, df_raw):
        """
        Pull the raw realized-variance series out of the file.

        Priority: the --target column if it is present, then 'RV', then 'ln_RV',
        then the first numeric column with a warning -- silently transforming the
        wrong column is the failure mode worth being loud about.  A column named
        ln_RV / log_RV is exponentiated once, because the forward windows have to
        be averaged on the variance scale before the log is taken.
        """
        cols = [c for c in df_raw.columns if c != 'date']
        pick = None
        for cand in (self.target, 'RV', 'ln_RV'):
            if cand in cols:
                pick = cand
                break
        if pick is None:
            pick = df_raw[cols].select_dtypes('number').columns[0]
            print("  WARNING: no '{}'/'RV'/'ln_RV' column. Using '{}' AS RAW RV."
                  .format(self.target, pick))
        s = df_raw[pick].astype(float)
        if pick.lower() in ('ln_rv', 'log_rv'):
            print("  Column '{}' found -> already log scale, exp() once to raw RV."
                  .format(pick))
            s = np.exp(s)
        else:
            print("  Column '{}' found -> raw realized variance.".format(pick))
        return s

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if 'date' not in df_raw.columns:
            raise ValueError("Dataset_RV needs a 'date' column in {}."
                             .format(self.data_path))

        dates = pd.to_datetime(df_raw['date'])
        rv = pd.Series(self.__read_rv__(df_raw).values, index=dates)
        rv = rv.sort_index().dropna()

        n_bad = int((rv <= 0).sum())
        if n_bad:
            print('  Dropped {} non-positive RV row(s) (non-trading days).'
                  .format(n_bad))
            rv = rv[rv > 0]

        h = self.horizon
        # Input channel: ln(RV).  Target: ln of the ARITHMETIC forward mean --
        # averaged on the raw variance scale, logged only afterwards.  Indexing
        # is such that y_agg[t] covers rows t .. t+h-1, so the window ending at
        # row t-1 forecasts it.
        ln_rv = np.log(rv)
        fwd = rv.rolling(h).mean().shift(-(h - 1))
        y_agg = np.log(fwd)

        # Calendar split, from data_provider/splits.py -- the same module HAR-RV
        # reads, so the test window is identical for both families.
        cal = SPLITS[self.asset]
        lo, hi = dl_bounds(cal, self.flag)
        first, last = row_range(rv.index, lo, hi)
        # The forecast origins ARE the rows of the split; the seq_len rows in
        # front of it are look-back only, and are inputs at every origin the way
        # HAR's 22-day rolling window is. Rows outside [lo, hi] are never a
        # target here, which is what embargoes the seam with the next split.
        border1 = max(0, first - self.seq_len)
        border2 = last + 1
        warmup = self.seq_len - (first - border1)
        print('  {:5s} {} .. {}  ({} rows{})'.format(
            self.flag, fmt_month(lo), fmt_month(hi), last - first + 1,
            '' if warmup <= 0 else
            '; first {} unforecastable, no look-back before them'.format(warmup)))

        x_all = ln_rv.values.reshape(-1, 1)
        y_all = y_agg.values.reshape(-1, 1)   # trailing h-1 rows are NaN
        if self.scale:
            # Training ln(RV) rows only -- never the validation or test window,
            # and read off the calendar rather than this split's own slice, so
            # all three splits share one set of statistics.
            t0, t1 = row_range(rv.index, *dl_bounds(cal, 'train'))
            self.scaler.fit(x_all[t0:t1 + 1])
            self.mean_ = float(self.scaler.mean_[0])
            self.std_ = float(self.scaler.scale_[0])
        else:
            self.mean_, self.std_ = 0.0, 1.0
        # Applied by hand rather than through scaler.transform so the NaN tail of
        # the target passes through untouched; those rows are never enumerated.
        x_all = (x_all - self.mean_) / self.std_
        y_all = (y_all - self.mean_) / self.std_

        df_stamp = pd.DataFrame({'date': rv.index[border1:border2]})
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values),
                                       freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = x_all[border1:border2]
        self.data_y = y_all[border1:border2]
        self.data_stamp = data_stamp
        self.dates = rv.index[border1:border2]
        # The FULL trading-day index and this split's slice of it. Dataset_RV
        # itself never needs them, but a subclass that carries a second series
        # alongside RV (Dataset_RV_Events) has to align that series to exactly
        # the rows kept here -- after the sort and the non-positive drop -- and
        # to fit its own scaler on the training rows, which are not inside this
        # split's slice when the split is validation or test.
        self.full_index = rv.index
        self.borders = (border1, border2)
        # Raw RV over the split, kept for descriptive output and the QLIKE floor.
        self.rv_raw = rv.values[border1:border2]

        if len(self) <= 0:
            raise ValueError(
                'The {} split holds {} rows, too few for seq_len={} + h={}.'
                .format(self.flag, border2 - border1, self.seq_len, h))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len

        seq_x = self.data_x[s_begin:s_end]
        # label_len rows of context (0 for ModernTCN) followed by the ONE
        # aggregated target for rows s_end .. s_end+h-1.
        seq_y = np.concatenate([self.data_x[r_begin:s_end],
                                self.data_y[s_end:s_end + 1]], axis=0)
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_begin + self.label_len + 1]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # One window per usable origin: the target must close inside the split,
        # which is what embargoes the seam with the next split.
        return len(self.data_x) - self.seq_len - self.horizon + 1

    def target_dates(self):
        """Opening date of each window's h-day target, in window order."""
        first = self.seq_len
        return self.dates[first:first + len(self)]

    def inverse_transform(self, data):
        """Standardised units -> ln(RV).  Not raw RV: exp() is a separate step."""
        return np.asarray(data) * self.std_ + self.mean_


class Dataset_RV_Events(Dataset_RV):
    """
    Dataset_RV plus the macro news-event calendar -- the EventTCN loader.

    Everything about the RV series is inherited unchanged: the AGGREGATED target
    on the LOG scale, log-of-mean rather than mean-of-logs, the calendar splits
    from data_provider/splits.py, the training-only standardisation.  This class
    only bolts a second, purely exogenous matrix onto the same rows.

        Y_t^(h) = ln( (1/h) * Sum_{k=1..h} RV_{t+k} )      <- unchanged

    Note this is NOT the target the FilmTCN repo's event model uses.  There the
    aggregation is a mean of ln(RV) -- the log of a GEOMETRIC forward mean --
    so its losses are computed on a smaller, much smoother object and do not sit
    beside these ones, or beside HAR-RV, at any horizon past h=1.  The event
    conditioning ports over; the numbers do not.

    __getitem__ returns two extra tensors:

        seq_x_events : (seq_len, F)  the calendar over the LOOK-BACK rows
                                     [t-seq_len+1 .. t] -- a past covariate,
                                     embedded and injected at the stem.
        seq_y_events : (h, F)        the calendar over the TARGET window
                                     [t+1 .. t+h] -- a future covariate, pooled
                                     into the FiLM generator.

    The horizon slice is exactly the h days whose mean RV is being forecast, and
    it holds only what was on the release calendar: that an event is scheduled,
    its currency and its vendor impact rating, never its outcome.  The forecast
    origin is unchanged, so this is a known-in-advance covariate, not a leak.

    Feature construction, the role vocabulary and the dead-column rule live in
    data_provider/calendar_events.py.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='realized_volatility.csv',
                 target='RV', scale=True, timeenc=0, freq='d', asset=None,
                 event_path='eurusd_calendar_events_2010_2025 (1).csv',
                 event_vocab='role'):
        self.event_path = event_path
        self.event_vocab = event_vocab
        super().__init__(root_path=root_path, flag=flag, size=size,
                         features=features, data_path=data_path, target=target,
                         scale=scale, timeenc=timeenc, freq=freq, asset=asset)

    def __read_data__(self):
        super().__read_data__()

        frame, count_cols = build_event_frame(
            os.path.join(self.root_path, self.event_path),
            vocab=self.event_vocab, verbose=(self.flag == 'train'))

        # Align to the trading days Dataset_RV actually kept -- after its sort
        # and its non-positive drop -- so event row i and RV row i are the same
        # day by construction rather than by luck.
        aligned = align_to_index(frame, self.full_index)
        values = aligned.values.astype(np.float32)
        columns = list(aligned.columns)

        # Training rows on the split calendar, NOT this split's own slice: the
        # scaler and the dead-column rule must read the same window whichever
        # split is being built, or the three would disagree on the feature set.
        cal = SPLITS[self.asset]
        t0, t1 = row_range(self.full_index, *dl_bounds(cal, 'train'))
        values, columns = drop_dead_columns(
            values, columns, (t0, t1 + 1), verbose=(self.flag == 'train'))

        # Counts are standardised on those training rows; the evt_* columns stay
        # raw small integers, which is already the scale an embedding wants.
        count_idx = [j for j, c in enumerate(columns) if c in set(count_cols)]
        if count_idx and self.scale:
            block = values[t0:t1 + 1][:, count_idx]
            mean = block.mean(axis=0)
            std = block.std(axis=0)
            std[std < 1e-8] = 1.0
            values[:, count_idx] = (values[:, count_idx] - mean) / std

        border1, border2 = self.borders
        self.event_cols = columns
        self.n_event_features = len(columns)
        self.data_events = values[border1:border2]

        # A window whose horizon reaches past the end of the calendar would be
        # told, falsely, that nothing is scheduled. Warn rather than fail: the
        # run is still valid, the last few origins are just uninformed.
        last_event_day = frame.index.max()
        last_needed = self.dates[-1]
        if last_needed > last_event_day:
            n = int((self.dates > last_event_day).sum())
            print('  WARNING: the calendar stops at {} but the {} split runs to '
                  '{}; {} row(s) carry an all-zero schedule they did not earn.'
                  .format(last_event_day.date(), self.flag,
                          last_needed.date(), n))

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = super().__getitem__(index)

        s_end = index + self.seq_len
        seq_x_events = self.data_events[index:s_end]
        # Exactly the h rows the aggregated target averages over: data_y[s_end]
        # covers rows s_end .. s_end+h-1, and __len__ guarantees they exist.
        seq_y_events = self.data_events[s_end:s_end + self.horizon]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_events, seq_y_events
