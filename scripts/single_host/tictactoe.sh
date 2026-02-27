#!/bin/bash
# Launch TTT-Discover training for tic-tac-toe evaluation polynomial
#
# Smoke test (~20 min):   ./scripts/single_host/tictactoe.sh
# Production (hours):     GROUPS_PER_BATCH=64 GROUP_SIZE=8 NUM_EPOCHS=50 ./scripts/single_host/tictactoe.sh
set -euo pipefail

if [ -z "${TINKER_API_KEY:-}" ]; then
    echo "ERROR: TINKER_API_KEY not set. Run: export TINKER_API_KEY='tml-...'"
    exit 1
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"

cd /tmp/discover

echo "=== Starting Tic-Tac-Toe TTT-Discover training ==="
echo "  Objective: minimize MSE of degree-3 multilinear polynomial vs minimax"
echo "  Groups per batch: ${GROUPS_PER_BATCH:-2}"
echo "  Group size: ${GROUP_SIZE:-2}"
echo "  Epochs: ${NUM_EPOCHS:-3}"
echo ""

python -m tinker_cookbook.recipes.ttt.train \
    env=tictactoe \
    problem_idx=tictactoe_default \
    model_name=openai/gpt-oss-120b \
    renderer_name=gpt_oss_high_reasoning \
    sampler_type=puct \
    initial_exp_type=random \
    num_cpus_per_task=1 \
    eval_timeout=120 \
    groups_per_batch="${GROUPS_PER_BATCH:-2}" \
    group_size="${GROUP_SIZE:-2}" \
    num_epochs="${NUM_EPOCHS:-3}" \
    learning_rate=4e-5 \
    lora_rank=32 \
    max_tokens=26000 \
    adv_estimator=entropic \
    loss_fn=importance_sampling \
    save_every=1 \
    eval_every=0 \
    test_num_rollouts=0 \
    log_path="./logs/tictactoe-$(date +%Y%m%d-%H%M%S)" \
    "$@"
