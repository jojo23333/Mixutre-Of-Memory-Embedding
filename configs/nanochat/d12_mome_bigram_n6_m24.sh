#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

exec "${PYTHON:-python}" -m scripts.run \
  --name nanochat/d12_mome_bigram_n6_m24 \
  --training-shards 100 \
  --seeds 42,43 \
  --nproc-per-node 4 \
  --attention-backend auto \
  "$@" -- \
  --depth 12 \
  --aspect-ratio 64 \
  --head-dim 128 \
  --max-seq-len 2048 \
  --window-pattern L \
  --target-flops 3e18 \
  --device-batch-size 32 \
  --total-batch-size 524288 \
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
  --eval-every -1 \
  --eval-tokens 20971520 \
  --core-metric-every -1 \
  --skip-checkpoint \
  --model-type gpt \
  --moe-ve \
  --moe-ve-setting 0_2_24 \
  --moe-ve-gate-nl sigmoid-norm \
  --moe-ve-router-input hidden \
  --moe-ve-slot-mapping none \
  --moe-ve-slot-factor 1 \
  --moe-ve-balance-lr 0 \
  --moe-ve-bias-scope slot \
  --moe-ve-bias-update none \
  --moe-ve-bias-min-visits 0 \
  --moe-ve-bias-powerlaw-n 1.4 \
  --moe-ve-maxvio-window 0 \
  --value-embedding-lr 0.2 \
  --value-embedding-adam-beta1 0.8 \
  --value-embedding-adam-beta2 0.995 \
  --value-embedding-adam-eps 1e-10 \
  --value-embedding-weight-decay 0.001 \
  --moe-ve-network-shared-table \
  --moe-ve-slot-index-mode bigram \
  --moe-ve-bigram-slot-factor 6 \
  --final-eval-splits train,val
