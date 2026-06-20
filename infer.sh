python scripts/infer/vllm_rollout.py \
  --input-parquet ${DATA_DIR}/OpenThoughts3_opd.parquet \
  --model-path ${MODEL_DIR}/Qwen3-4B \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --enable-thinking false \
  --enable-rejection-sampling true \
  --max-attempts-per-rollout 3