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
# ==============================================================================

seq_len=336

for pred_len in 1 5 22
do
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
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 31 \
  --small_size 5 \
  --dims 32 32 32 32 \
  --dw_dims 32 32 32 32 \
  --head_dropout 0.0 \
  --dropout 0.3 \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 64 \
  --patience 10 \
  --learning_rate 0.0005 \
  --des Exp \
  --lradj type3 \
  --use_multi_scale False \
  --small_kernel_merged False
done
