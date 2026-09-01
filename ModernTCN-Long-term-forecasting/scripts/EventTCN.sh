# ==============================================================================
# EventTCN -- ModernTCN on realized volatility, CONDITIONED on the macro news
# calendar.  Same aggregated horizon target and same log scale as ./scripts/RV.sh:
#
#   input   x_t     =     ln( RV_t )
#   target  Y_t^(h) = ln( (1/h) * Sum_{k=1}^{h} RV_{t+k} )
#
# so --pred_len is the horizon h and the head emits ONE number, the h-day mean.
# LOG OF MEAN, not mean of logs -- see data_provider/data_loader.py.  This is
# deliberately NOT the target the FilmTCN repo's event model uses: there the
# aggregation is a mean of ln(RV), the log of a geometric forward mean, so its
# losses are a different object and cannot be read next to these or next to
# HAR-RV at any horizon past h=1.  The conditioning ports over; the numbers do not.
#
# ------------------------------------------------------------------------------
# WHAT THE CONDITIONING IS
# ------------------------------------------------------------------------------
# data/eurusd_calendar_events_2010_2025 (1).csv holds the EUR/USD economic
# calendar: a time stamp, a currency, an impact rating and a name per scheduled
# event.  No actual, no forecast, no previous -- only that an event is SCHEDULED.
# data_provider/calendar_events.py turns it into per-day counts, and the loader
# hands the model two slices:
#
#   PAST    the look-back window's calendar, embedded and fed through its own
#           patch stem into the backbone (--event_past)
#   FUTURE  the horizon's known schedule, pooled and used to generate a FiLM
#           (gamma, beta) over the final feature map (--event_future)
#
# The future slice is a genuine known-in-advance covariate: it says an FOMC
# statement lands inside the window being forecast, never what it said.  The FiLM
# generator is zero-initialised, so that path is an exact identity at step 0.
#
# ------------------------------------------------------------------------------
# THE CONTROL
# ------------------------------------------------------------------------------
# --data RV and --data RVEvents differ ONLY in what the loader hands the model:
# same rows, same splits, same target, same scaler.  So the ablation is one flag:
#
#   sh ./scripts/RV.sh "1 5 22"        # unconditioned control
#   sh ./scripts/EventTCN.sh "1 5 22"  # conditioned
#
# The backbone hyperparameters below are RV.sh's, i.e. tuned for the
# unconditioned model.  That is the right CONTROL -- identical backbone, events
# the only difference -- but it is not a tuned EventTCN.  Re-running the Optuna
# search over --data RVEvents is the obvious next step, and until then a null
# result here means "events do not help THIS backbone", not "events do not help".
#
# Expect the gain to shrink with the horizon.  Over this calendar the forward
# high-impact event count has a coefficient of variation of 0.92 at h=1, 0.50 at
# h=5 and 0.27 at h=22: by the monthly horizon almost every window looks alike,
# so there is very little left to condition on.
#
#   sh ./scripts/EventTCN.sh              # h = 1 only (default)
#   sh ./scripts/EventTCN.sh "1 5 22"     # all three horizons
#
# Ablation switches worth running once each (h=1, where the signal is):
#   --event_past False                 future-only: FiLM alone, identity at init
#   --event_future False               past-only: no future covariate at all
#   --event_fusion channel             past events as a backbone variable
#   --event_vocab name                 one column per raw event name, not per role
#   --event_dim 4 | 8 | 16             width of the event embedding
# ==============================================================================

HORIZONS="${1:-1}"

for pred_len in $HORIZONS
do

case $pred_len in
  1)   seq_len=22; patch_size=32; patch_stride=8; ffn_ratio=3; large_size=51
       small_size=3;  num_blocks=3; dropout=0.4155; head_dropout=0.2993
       learning_rate=0.00550; batch_size=128 ;;
  5)   seq_len=35; patch_size=32; patch_stride=2; ffn_ratio=3; large_size=13
       small_size=5;  num_blocks=3; dropout=0.0751; head_dropout=0.3690
       learning_rate=0.00431; batch_size=128 ;;
  22)  seq_len=35; patch_size=4;  patch_stride=8; ffn_ratio=2; large_size=31
       small_size=5;  num_blocks=1; dropout=0.2332; head_dropout=0.3228
       learning_rate=0.00779; batch_size=256 ;;
  *)   echo "No tuned configuration for h=$pred_len -- searched horizons are 1, 5 and 22."
       continue ;;
esac

python -u run.py \
  --is_training 1 \
  --model_id EventTCN_$seq_len'_'$pred_len \
  --model ModernTCN \
  --data RVEvents \
  --root_path ./data/ \
  --data_path realized_volatility.csv \
  --event_data_path 'eurusd_calendar_events_2010_2025 (1).csv' \
  --event_vocab role \
  --event_dim 8 \
  --event_past True \
  --event_future True \
  --event_fusion inject \
  --features S \
  --target RV \
  --asset forex \
  --freq d \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --enc_in 1 \
  --ffn_ratio $ffn_ratio \
  --patch_size $patch_size \
  --patch_stride $patch_stride \
  --num_blocks $num_blocks \
  --large_size $large_size \
  --small_size $small_size \
  --dims 32 32 32 32 \
  --dw_dims 32 32 32 32 \
  --dropout $dropout \
  --head_dropout $head_dropout \
  --learning_rate $learning_rate \
  --batch_size $batch_size \
  --revin 1 \
  --random_seed 2021 \
  --itr 5 \
  --train_epochs 50 \
  --patience 10 \
  --des Exp \
  --lradj type3 \
  --use_multi_scale False \
  --small_kernel_merged False
done
