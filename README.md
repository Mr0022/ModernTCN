# ModernTCN (ICLR 2024 Spotlight)
This is an official implementation of paper: [ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis](https://openreview.net/forum?id=vpJMJerXHU#).

## Our Paper
Donghao Luo and Xue Wang. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis. In International Conference on Learning Representations, 2024.
[[Our paper in OpenReview]](https://openreview.net/forum?id=vpJMJerXHU#).

We study the open question of how to better use convolution in time series analysis and we take a seldom-explored way in time series community to successfully bring convolution back to time series analysis.

As a pure convolution structure, our ModernTCN achieves the consistent state-of-the-art performance on five mainstream time series analysis tasks (long-term and short-term forecasting, imputation, classification and anomaly detection) while maintaining the efficiency advantage of convolution-based models, therefore providing a better balance of efficiency and performance.

## ModernTCN Block

**ModernTCN block design:** 
ModernTCN block can achieve larger ERF and better capture the cross-variable dependency, therefore being more suitable for time series analysis.

|![image](fig/fig_block.png) | ![image](fig/fig_erf.png)
|:--:|:--:|
| *Figure 1. ModernTCN block design.* | *Figure 2. Visualization of ERF.* |

## Main Results

**Main Results:** 
Our ModernTCN achieves the consistent state-of-the-art performance on five mainstream time series analysis tasks with better efficiency.
![Block Design](fig/fig_mainresult.png)
## Get Started

1. Install Python 3.7 and necessary dependencies.
```
pip install -r requirements.txt
```
2. Download data. You can obtain all datasets from [[Times-series-library](https://github.com/thuml/Time-Series-Library)].

3. Long-term forecasting tasks.
 
We provide the long-term forecasting experiment coding in `./ModernTCN-Long-term-forecasting` and experiment scripts can be found under the folder `./scripts`. To run the code on ETTh2, just run the following command:

```
cd ./ModernTCN-Long-term-forecasting

sh ./scripts/ETTh2.sh
```

**Realized volatility (`--data RV`) - aggregated horizon, log scale.**

`--data RV` forecasts realized volatility the way `HAR-RV_RUN.PY --log` does, so
the two can be compared row for row. It differs from the standard setup in two
ways, and does both unconditionally:

* **Aggregated target.** The model emits ONE number per window - the h-day
  average - not an h-step path. `--pred_len` is therefore the HORIZON h
  (1 daily, 5 weekly, 22 monthly) and the head's output length is 1. At h=5 the
  model predicts the mean of the next five days, once; it is never asked for the
  five individual days.
* **Log scale, log of mean.** The input channel is `ln(RV)` and the target is the
  log of the *arithmetic* forward mean, i.e. the log sits OUTSIDE the sum:

  ```
  x_t     =     ln( RV_t )
  Y_t^(h) = ln( (1/h) * Sum_{k=1..h} RV_{t+k} )
  ```

  Log-of-mean, not mean-of-logs: `(1/h) * Sum ln RV` would be the log of a
  geometric mean, a smaller and much smoother object. With log-of-mean,
  `exp(Y_t^(h))` is precisely the raw h-day average HAR-RV predicts, at every
  horizon. Sum vs mean is cosmetic here - `ln(sum) = ln(mean) + ln(h)`, a
  constant the model absorbs - and the mean keeps losses on one scale across
  horizons. At h=1 the whole thing collapses to plain next-day `ln(RV)`.

The CSV needs a `date` column plus `RV` (raw variance) or `ln_RV`. Non-positive
rows are dropped, as in HAR-RV. Standardisation is fitted on the training
`ln(RV)` rows only and applied to the target after it is built and logged.
Window enumeration embargoes the split seams, so no training target reaches into
validation rows and no validation target into test.

**Splits.** Both families read their calendar from `data_provider/splits.py`,
which is the single source of truth for where the boundaries fall
(`--asset`, default `forex`):

| | train | validation | test |
|---|---|---|---|
| ModernTCN | 2010-01-01 - 2021-12-31 | 2022-01-01 - 2023-12-31 | 2024-01-01 - 2025-04-07 |
| HAR-RV | 2010-01-01 - 2023-12-31 (train + val) | - | 2024-01-01 - 2025-04-07 |

HAR-RV is OLS with nothing to tune, so it folds the validation window into its
estimation sample and keeps the same test window. The two therefore forecast
*identical* test rows - 328, 324 and 307 origins at h = 1, 5, 22, the h-1
shrinkage being the embargo. Rows after 2025-04-07 fall outside every window and
are unused. The one place the families cannot match is the start of training:
HAR loses 22 rows to its monthly component and ModernTCN loses `seq_len`, since
neither can look back past the first row in the file.

**Hyperparameters.** One Bayesian-optimisation search per horizon. The h=1
column is also the argparse default in `run.py`, so a run that omits these flags
reproduces the 1-step-ahead setup:

| | h=1 (default) | h=5 | h=22 |
|---|---|---|---|
| `seq_len` | 22 | 35 | 35 |
| `patch_size` | 32 | 32 | 4 |
| `patch_stride` | 8 | 2 | 8 |
| `dims` | 32 | 32 | 32 |
| `ffn_ratio` | 3 | 3 | 2 |
| `large_size` | 51 | 13 | 31 |
| `small_size` | 3 | 5 | 5 |
| `num_blocks` | 3 | 3 | 1 |
| `dropout` | 0.4155 | 0.0751 | 0.2332 |
| `head_dropout` | 0.2993 | 0.3690 | 0.3228 |
| `learning_rate` | 0.00550 | 0.00431 | 0.00779 |
| `batch_size` | 128 | 128 | 256 |
| `revin` | 1 | 1 | 1 |
| best objective | 0.17672 | 0.08441 | 0.06684 |

`scripts/RV.sh` carries all three; it runs h=1 alone by default.

```
cd ./ModernTCN-Long-term-forecasting

sh ./scripts/RV.sh              # h = 1 (default)
sh ./scripts/RV.sh "1 5 22"     # all three horizons

python HAR-RV_RUN.PY --data data/realized_volatility.csv --log --asset forex
```

**Seeds.** `--itr` is the number of seeds, not repeats of one run: iteration
`i` seeds everything with `random_seed + i` before the model is built, so
`--random_seed 2021 --itr 5` covers 2021-2025. MSE, MAE and QLIKE are printed
after every seed and averaged at the end, with the standard deviation beside the
mean - one seed is a draw, not a result:

```
  SEED SUMMARY  --  RV_22_1  |  5 seed(s): 2021, 2022, 2023, 2024, 2025  |  h=1
  --------------------------------------------------------------------------
  seed           MSE[ln]       MAE[ln]     QLIKE[RV]        MSE_RV     MAE_RV
  --------------------------------------------------------------------------
  2021          0.309805      0.421181      0.191580      0.026944   0.087085
  ...
  mean          0.312448      0.421338      0.192328      0.026773   0.088801
  std           0.007336      0.004760      0.005482      0.000384   0.001428
```

The table also lands in `results/<model_id>_<des>_seed_metrics.csv`.

Each run additionally writes `results/<setting>/rv_metrics.csv` and
`rv_forecasts.csv`. The metrics restate the forecasts on the two scales HAR-RV
reports - MSE/MAE in `ln(RV)` units, and QLIKE / MSE_RV / MAE_RV after a
Jensen-corrected `exp()`, `E[RV|F] = exp(E[ln RV|F] + sigma^2/2)`, with
`sigma^2` taken from the validation log residuals - under the same column names
`har_rv_log_all_metrics.csv` uses, so the two files concatenate directly.
QLIKE (Patton, 2011) lives in `utils/metrics.py` and is shared with HAR-RV; it
is asymmetric, punishing an under-forecast variance far harder than an
over-forecast one, which is why it is reported next to MSE rather than instead
of it.

4. Short-term forecasting tasks.

We provide the short-term forecasting experiment coding in `./ModernTCN-short-term` and experiment scripts can be found under the folder `./scripts`. Please run the following command:

```
cd ./ModernTCN-short-term

sh ./scripts/M4.sh
```

5. Imputation tasks.

We provide the imputation experiment coding in `./ModernTCN-imputation` and experiment scripts can be found under the folder `./scripts`. To run the code on ETTh2, just run the following command:

```
cd ./ModernTCN-imputation

sh ./scripts/ETTh2.sh
```

6. Classification tasks.

We provide the classification experiment coding in `./ModernTCN-classification` and experiment scripts can be found under the folder `./scripts`. Please run the following command:

```
cd ./ModernTCN-classification

sh ./scripts/classification.sh
```

7. Anomaly detection tasks.

We provide the anomaly detection experiment coding in `./ModernTCN-detection` and experiment scripts can be found under the folder `./scripts`. To run the code on SWaT, just run the following command:

```
cd ./ModernTCN-detection

sh ./scripts/SWaT.sh
```

## Contact
If you have any question or want to use the code, please contact [ldh21@mails.tsinghua.edu.cn](mailto:ldh21@mails.tsinghua.edu.cn).

## Citation

If you find this repo useful, please cite our paper. 
```
@inproceedings{
donghao2024moderntcn,
title={Modern{TCN}: A Modern Pure Convolution Structure for General Time Series Analysis},
author={Luo donghao and wang xue},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=vpJMJerXHU}
}
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/ts-kim/RevIN

https://github.com/PatchTST/PatchTST

https://github.com/thuml/Time-Series-Library

https://github.com/facebookresearch/ConvNeXt

https://github.com/MegEngine/RepLKNet

