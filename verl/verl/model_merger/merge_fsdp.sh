#!/bin/bash

set -e  # 任意报错直接退出

# ===============================
# 只需要修改这一行
# ===============================
RUN_PATH=/ossfs/workspace/aitech_aidata/chuwei/ckpt/OPD/ckpts/token_reward_direct_Formal-DAPO-Math-17k-opd-top0_lr1e-5_Phi_Qwen3-1.7B-Base_Phi-4-mini-reasoning_16384-T_1.0-Tch_1.0-n_2-mbs_8-topk_0-topk_strategy_only_stu-rw_student_p-2026-06-19_20-19-57-OPD/global_step_2





# LOCAL_DIR="${BASE_DIR}/${RUN_PATH}/actor"
# TARGET_DIR="${BASE_DIR}/${RUN_PATH}_merge_hf"

LOCAL_DIR="${RUN_PATH}/actor"
TARGET_DIR="${RUN_PATH}_merge_hf"

echo "======================================"
echo "Merging model..."
echo "Source: ${LOCAL_DIR}"
echo "Target: ${TARGET_DIR}"
echo "======================================"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${LOCAL_DIR}" \
    --target_dir "${TARGET_DIR}"

echo "======================================"
echo "Merge completed successfully."
echo "Saved to: ${TARGET_DIR}"
echo "======================================"

# ===============================
# 删除 LOCAL_DIR
# ===============================
if [ -d "${LOCAL_DIR}" ]; then
    echo "Deleting ${LOCAL_DIR} ..."
    rm -rf "${LOCAL_DIR}"
    echo "Deleted ${LOCAL_DIR}"
else
    echo "Warning: ${LOCAL_DIR} not found, skip deletion."
fi