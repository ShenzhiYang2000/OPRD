python scripts/infer/vllm_rollout.py \
  --input-parquet /ossfs/workspace/aml0/484999/code/OPD-main/datasets/OpenThoughts3_opd.parquet \
  --model-path /ossfs/workspace/aml0/Qwen3/Qwen3-4B \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --enable-thinking false \
  --enable-rejection-sampling true \
  --max-attempts-per-rollout 3