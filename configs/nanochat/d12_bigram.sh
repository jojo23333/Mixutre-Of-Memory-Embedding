#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

exec "${PYTHON:-python}" -m scripts.run \
  --name nanochat/d12_bigram \
  --training-shards 100 \
  --seeds 42,43,44 \
  --nproc-per-node 4 \
  --attention-backend sdpa \
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
  --model-type bigram_engram_gpt \
  --bigram-engram-dict-size 6
