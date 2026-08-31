import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def QLIKE(pred, true, floor=None):
    """
    QLIKE loss (Patton, 2011) for volatility forecasts. Smaller is better.

        QLIKE = mean( RV/RV_hat - ln(RV/RV_hat) - 1 )

    Both arguments must be VARIANCES, not logs and not standardised values --
    the ratio and its log are only defined on the positive half line. A caller
    working on the log scale exponentiates first (see Exp_Main._report_rv, which
    also applies the lognormal Jensen correction on the way).

    Unlike MSE, QLIKE is asymmetric: it punishes under-prediction of a variance
    far harder than over-prediction, which is the behaviour a risk application
    wants and the reason it is reported alongside MSE rather than instead of it.

    A forecast <= 0 has no QLIKE, so it is FLOORED rather than dropped --
    dropping would quietly remove the worst forecasts and flatter the score.
    `floor` defaults to 1e-4 * mean(actual). Rows with a non-positive ACTUAL are
    skipped instead: there the loss is undefined for the observation, not for
    the forecast. Count the floored forecasts yourself if the number matters --
    a large count means the score leans on the floor.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    valid = true > 0
    if not np.any(valid):
        return float('nan')
    if floor is None:
        floor = 1e-4 * float(np.mean(true[valid]))
    safe = np.where(pred <= 0, floor, pred)
    ratio = true[valid] / safe[valid]
    return float(np.mean(ratio - np.log(ratio) - 1))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr
