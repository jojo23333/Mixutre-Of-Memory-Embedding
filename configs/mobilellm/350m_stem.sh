#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

exec "${PYTHON:-python}" -m scripts.run \
  --name mobilellm/350m_stem \
  --training-shards 520 \
  --seeds 42 \
  --nproc-per-node 4 \
  --attention-backend sdpa \
  "$@" -- \
  --model-type stemgpt_350m \
  --total-batch-size 524288 \
  --max-seq-len 2048 \
  --depth 32 \
  --aspect-ratio 30 \
  --model-dim 960 \
  --head-dim 64 \
  --num-kv-heads 5 \
  --window-pattern L \
  --num-iterations 38146 \
  --embedding-lr 0.3 \
  --value-embedding-lr 0.1 \
  --unembedding-lr 0.004 \
  --weight-decay 0.2 \
  --matrix-lr 0.02 \
  --scalar-lr 0.5 \
  --adam-beta1 0.8 \
  --adam-beta2 0.95 \
  --warmup-ratio 0 \
  --warmdown-ratio 0.4 \
  --final-lr-frac 0 \
  --eval-tokens 2097152 \
  --save-every 1907 \
  --core-metric-every -1 \
  --sample-every -1 \
  --final-eval-splits train,val \
  --skip-checkpoint \
  --device-batch-size 8 \
  --eval-every 7629 \
  --stem-layers 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31
