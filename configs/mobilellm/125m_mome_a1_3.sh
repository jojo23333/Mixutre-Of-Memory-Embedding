#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

exec "${PYTHON:-python}" -m scripts.run \
  --name mobilellm/125m_mome_a1_3 \
  --training-shards 160 \
  --seeds 42 \
  --nproc-per-node 4 \
  --attention-backend sdpa \
  "$@" -- \
  --model-type stemgpt_350m \
  --total-batch-size 524288 \
  --max-seq-len 2048 \
  --depth 30 \
  --aspect-ratio 19 \
  --model-dim 576 \
  --head-dim 64 \
  --num-kv-heads 3 \
  --weight-tying \
  --window-pattern L \
  --num-iterations 9536 \
  --embedding-lr 0.3 \
  --unembedding-lr 0.004 \
  --weight-decay 0.2 \
  --matrix-lr 0.02 \
  --scalar-lr 0.5 \
  --adam-beta1 0.8 \
  --adam-beta2 0.95 \
  --warmup-ratio 0 \
  --warmdown-ratio 0.4 \
  --final-lr-frac 0 \
  --eval-every 1907 \
  --eval-tokens 2097152 \
  --save-every 1907 \
  --core-metric-every -1 \
  --sample-every -1 \
  --skip-checkpoint \
  --final-eval-splits train,val \
  --device-batch-size 32 \
  --stem-layers none \
  --moe-ve \
  --value-embeds-layers 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29 \
  --moe-ve-setting 0_1_3 \
  --moe-ve-gate-nl softmax \
  --moe-ve-router-input value \
  --moe-ve-balance-lr 0.001 \
  --moe-ve-bias-scope slot \
  --moe-ve-bias-update none \
  --moe-ve-bias-min-visits 0 \
  --moe-ve-bias-powerlaw-n 1.4 \
  --moe-ve-slot-mapping none \
  --moe-ve-slot-factor 1 \
  --value-embedding-lr 0.1 \
  --value-embedding-weight-decay 0.001
