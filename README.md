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

**Colab.** [`RV_benchmark_colab.ipynb`](RV_benchmark_colab.ipynb) runs both families over all
three horizons and prints the comparison as a table and as LaTeX:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mr0022/ModernTCNt/blob/claude/moderntcn-aggregation-log-ptj8ao/RV_benchmark_colab.ipynb)

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

**EventTCN (`--data RVEvents`) - conditioning on the macro news calendar.**

`--data RVEvents` is `--data RV` plus the EUR/USD economic calendar in
`data/eurusd_calendar_events_2010_2025 (1).csv`. Same rows, same splits, same
aggregated log-of-mean target, same scaler - the two differ *only* in what the
loader hands the model, so a pair of runs is a clean ablation.

The calendar carries no actual, forecast or previous value, only that an event
is **scheduled**, with its currency and the vendor's impact rating. That is what
makes the horizon slice a legitimate future covariate rather than a leak: the
model learns that an FOMC statement lands inside the window it is forecasting,
never what the statement said. Two paths, independently switchable:

* **Past events** (`--event_past`). The look-back window's calendar is embedded
  and patched by its own stem - same geometry as the value stem, separate
  weights - then either added onto the stage-0 feature map (`--event_fusion
  inject`) or carried as an extra variable through the whole backbone
  (`--event_fusion channel`), where ModernTCN's cross-variable ConvFFN mixes
  RV against events at every stage. It is dropped again before the head.
* **Future events** (`--event_future`). The horizon's known schedule is pooled
  into one vector and generates a FiLM `(gamma, beta)` pair over the final
  feature map: `x * (1 + gamma) + beta`. The generator is zero-initialised, so
  this path is an **exact identity at step 0** and only moves away from it where
  that reduces loss.

Neither path touches RevIN. RevIN normalises per window per channel, and the
event columns are sparse - 23% of trading days carry no scheduled event at all -
so a window with a constant event column would be divided by `sqrt(eps)`. Keeping
the event stream outside RevIN entirely is what makes `channel` fusion safe.

**Event features** are built by `data_provider/calendar_events.py`: per-day
counts (`n_events`, by impact, by currency) plus one column per (currency, event
key). `--event_vocab` picks what a key is:

* `role` (default) normalises names to roles - every `FOMC Member <person>
  Speaks` becomes one `cb_member_speech` column, Draghi / Lagarde / Trichet one
  `cb_chief_speech` column, all CPI prints one `cpi` column. 16 live event
  columns.
* `name` keeps one column per raw event name, ~90 after dead ones are dropped.

`role` is the default because officials churn and raw names churn with them.
Over this calendar **12 event names never occur in the training window yet fire
213 times in test** - 23.5% of test events - among them Barr, Collins, Cook,
Goolsbee, Jefferson, Kugler, Logan, Musalem, Schmid and Nagel. A column that is
identically zero in training receives no gradient at all, so its embedding keeps
its random initialisation and then injects a fixed random vector whenever it
fires out of sample. The mirror image is just as wasteful: Draghi (203 training
rows), Dudley (152) and Weidmann (132) each own a well-trained column that never
fires again after 2021. Role normalisation maps every dead name onto a column
with hundreds of training examples; whatever the vocabulary,
`drop_dead_columns()` then removes anything still identically zero over the
training rows, so no column can reach test time untrained.

```
cd ./ModernTCN-Long-term-forecasting

sh ./scripts/RV.sh       "1 5 22"     # unconditioned control
sh ./scripts/EventTCN.sh "1 5 22"     # conditioned

# dump the aligned feature matrix for inspection or for a HAR-X baseline
python data_provider/calendar_events.py \
  --calendar "data/eurusd_calendar_events_2010_2025 (1).csv" \
  --rv data/realized_volatility.csv --out data/events.csv
```

**Colab.** [`RV_eventtcn_comparison_colab.ipynb`](RV_eventtcn_comparison_colab.ipynb) runs the whole
comparison over 5 seeds and puts it in one table: HAR-RV, ModernTCN and EventTCN, each neural
family under **two** hyperparameter sets - this repo's `scripts/RV.sh` values and the FilmTCN
repo's plain-model Optuna studies - at all three horizons. The two arms within a set differ only
in whether the model sees the calendar, so each pair isolates the event effect:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mr0022/ModernTCNt/blob/claude/moderntcn-aggregation-log-ptj8ao/RV_eventtcn_comparison_colab.ipynb)

A note that notebook makes explicit: the hyperparameters `scripts/RV.sh` and the table above
present as this repo's per-horizon search are in fact the FilmTCN repo's **EventTCN** studies
(`tuningresults/EVENTTCN{1,5,22}`), matching to seven decimals on the objective and exactly on
every parameter. They were selected with an event-conditioned model in the loop and are reused
here for the unconditioned baseline.

`scripts/EventTCN.sh` reuses `RV.sh`'s backbone hyperparameters. That is the
right *control* - identical backbone, events the only difference - but it is not
a tuned EventTCN, so until `tune_optuna.py` is re-run over `--data RVEvents` a
null result means "events do not help this backbone", not "events do not help".

Expect the gain to fall away with the horizon. Over this calendar the forward
high-impact event count has a coefficient of variation of 0.92 at h=1, 0.50 at
h=5 and 0.27 at h=22, and no h=22 window is event-free at all: by the monthly
horizon almost every window looks alike, so there is very little left to
condition on. A large h=22 gain would be a reason to look for a leak, not to
celebrate.

Two things this port deliberately leaves for later. The horizon pool is a
**mean** over the h days, so a release on day 1 and one on day h condition the
model identically - a sum with learned per-position weights is the obvious next
experiment, and the likeliest reason h=22 shows nothing. And the honest
comparator for a gain here is **HAR-X** - HAR-RV with announcement counts as
extra regressors - not plain HAR-RV; without it a win shows only that the
calendar is informative, not that FiLM is the right way to use it.

**Hyperparameter search.** `tune_optuna.py` runs Bayesian optimisation with a
Tree-structured Parzen Estimator (Optuna's `TPESampler`) over the space above,
scored on the **validation** split - the test window is never read during a
search:

```
pip install optuna
cd ./ModernTCN-Long-term-forecasting
python tune_optuna.py --n_trials 100                # h=1, the default
python tune_optuna.py --n_trials 100 --pred_len 5   # one search per horizon
```

`patch_stride <= patch_size` is enforced by pruning the one invalid pair rather
than resampling from a shrinking list, which would hand TPE a differently-shaped
distribution in different trials. A `MedianPruner` abandons trials that fall
behind after a warm-up, and trials are written to a SQLite study as they finish,
so re-running the same command resumes rather than restarts. Results land in
`results_optuna/optuna_h{h}_best.json`, `_trials.csv` and `_command.sh` - the
last being the `run.py` invocation that re-runs the winner over 5 seeds.
`--objective {mse,mae,qlike}` picks the criterion; `mse`/`mae` are in ln(RV)
units, `qlike` on the back-transformed variance.

[`RV_optuna_search_colab.ipynb`](RV_optuna_search_colab.ipynb) runs the same search on Colab,
saving to Google Drive and producing a study that
[optuna-dashboard](https://optuna-dashboard.readthedocs.io/) opens directly:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mr0022/ModernTCNt/blob/claude/moderntcn-aggregation-log-ptj8ao/RV_optuna_search_colab.ipynb)

It keeps the working SQLite database on local disk and copies a consistent snapshot to Drive after
every trial, using SQLite's online-backup API. Drive's FUSE mount does not implement the file
locking SQLite needs, so a study written straight to `/content/drive/...` eventually fails with
*database is locked* and can be left corrupt rather than merely interrupted. With the snapshot, a
dropped runtime costs the trial in flight and nothing more - the notebook copies the study back and
continues.

**Sensitivity.** `sensitivity/ofat_sensitivity.py` anchors every hyperparameter at the tuned
configuration and sweeps one at a time over the grid Optuna searched, training each point over 5
seeds; `sensitivity/ofat_plots.py` turns the result into response curves, a tornado of the swing
either side of the anchor, and a ranking of which factors move the loss at all.

```
cd ./ModernTCN-Long-term-forecasting
python sensitivity/ofat_sensitivity.py --dry_run     # see the plan first
python sensitivity/ofat_sensitivity.py               # ~40 configs x 5 seeds
python sensitivity/ofat_plots.py
```

[`RV_sensitivity_colab.ipynb`](RV_sensitivity_colab.ipynb) runs the sweep on Colab, saving results
and figures to Google Drive as it goes:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mr0022/ModernTCNt/blob/claude/moderntcn-aggregation-log-ptj8ao/RV_sensitivity_colab.ipynb)

The anchor comes from `results_optuna/optuna_h1_best.json` when a search has been run, and falls
back to the tuned defaults otherwise. The sweep is resumable — points already in the CSV are
skipped. OFAT is *local* sensitivity: it says how the loss moves one step away from the optimum
and cannot see interactions, so read it alongside the Optuna study rather than instead of it.

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

