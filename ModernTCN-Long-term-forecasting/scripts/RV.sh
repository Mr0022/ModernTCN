# ==============================================================================
# ModernTCN on realized volatility -- AGGREGATED horizon target, LOG scale.
#
# --data RV always does both, exactly as HAR-RV_RUN.PY --log does:
#   input   x_t     =     ln( RV_t )
#   target  Y_t^(h) = ln( (1/h) * Sum_{k=1}^{h} RV_{t+k} )
#
# so --pred_len is the HORIZON h and the model emits ONE number per window --
# the h-day average -- not an h-step path. h = 1 daily, 5 weekly, 22 monthly,
# the three horizons HAR-RV reports.
#
# Splits come from data_provider/splits.py, the module HAR-RV_RUN.PY reads too:
#   train 2010-01-01..2021-12-31 | val 2022-01-01..2023-12-31
#   test  2024-01-01..2025-04-07
# HAR-RV folds validation into training and keeps that test window, so both
# families forecast identical rows. Run it over the same file with:
#
#   python HAR-RV_RUN.PY --data data/realized_volatility.csv --log --asset forex
#
# Losses land in ./results/<setting>/rv_metrics.csv with the column names
# HAR-RV_RUN.PY writes, so the two files concatenate directly.
#
# ------------------------------------------------------------------------------
# HYPERPARAMETERS -- Bayesian optimisation, one search per horizon
# ------------------------------------------------------------------------------
#                       h=1        h=5        h=22
#   seq_len             22         35         35
#   patch_size          32         32         4
#   patch_stride        8          2          8
#   dim                 32         32         32
#   ffn_ratio           3          3          2
#   large_size          51         13         31
#   small_size          3          5          5
#   num_blocks          3          3          1
#   dropout             0.4155     0.0751     0.2332
#   head_dropout        0.2993     0.3690     0.3228
#   learning_rate       0.00550    0.00431    0.00779
#   batch_size          128        128        256
#   revin               1          1          1
#   best objective      0.17672    0.08441    0.06684
#
# h=1 is the DEFAULT: its column is also the argparse defaults in run.py, so a
# run that omits these flags reproduces it. Every flag is still passed here so
# the three horizons read side by side.
#
#   sh ./scripts/RV.sh              # h = 1 only (default)
#   sh ./scripts/RV.sh "1 5 22"     # all three horizons
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
  --model_id RV_$seq_len'_'$pred_len \
  --model ModernTCN \
  --data RV \
  --root_path ./data/ \
  --data_path realized_volatility.csv \
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
  --itr 1 \
  --train_epochs 100 \
  --patience 10 \
  --des Exp \
  --lradj type3 \
  --use_multi_scale False \
  --small_kernel_merged False
done
