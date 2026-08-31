from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_RV
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'RV': Dataset_RV,
}

# Datasets whose target is an AGGREGATE of the forward window rather than a
# path: one number per window, so --pred_len is the horizon h and the model head
# emits a single step (run.py turns this into args.out_len).
AGG_DATA = ('RV',)


def is_aggregated(args) -> bool:
    return getattr(args, 'data', None) in AGG_DATA


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    agg = is_aggregated(args)

    if flag == 'test':
        shuffle_flag = False
        # Aggregated RV runs must score EVERY test window: dropping the last
        # partial batch would silently shorten the test sample and break the
        # row-for-row comparison against HAR-RV, which forecasts all of them.
        drop_last = not agg
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        # Validation of an aggregated run is read twice -- early stopping and the
        # residual variance behind the Jensen correction -- so it keeps every row
        # and a fixed order too.
        shuffle_flag = not (agg and flag == 'val')
        drop_last = not (agg and flag == 'val')
        batch_size = args.batch_size
        freq = args.freq

    kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    if agg:
        # Which calendar in data_provider/splits.py to split on. Only the
        # aggregated loaders take it; the others carry their own borders.
        kwargs['asset'] = getattr(args, 'asset', None)
    data_set = Data(**kwargs)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
