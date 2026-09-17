#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

exec "${PYTHON:-python}" -m scripts.run \
  --name qwen3/0p6b_vembedding \
  --training-shards 369 \
  --seeds 42 \
  --nproc-per-node 4 \
  --attention-backend auto \
  "$@" -- \
  --depth 28 \
  --aspect-ratio 64 \
  --head-dim 128 \
  --max-seq-len 2048 \
  --window-pattern L \
  --device-batch-size 8 \
  --total-batch-size 524288 \
  --embedding-lr 0.3 \
  --unembedding-lr 0.004 \
  --weight-decay 0.2 \
  --matrix-lr 0.02 \
  --scalar-lr 0.5 \
  --adam-beta1 0.8 \
  --adam-beta2 0.95 \
  --warmup-ratio 0.01 \
  --warmdown-ratio 0.4 \
  --final-lr-frac 0 \
  --eval-every 7629 \
  --eval-tokens 2097152 \
  --core-metric-every -1 \
  --skip-checkpoint \
  --model-dim 1024 \
  --num-heads 16 \
  --num-kv-heads 8 \
  --weight-tying \
  --num-iterations 38146 \
  --sample-every -1 \
  --final-eval-splits train,val \
  --model-type qwen3_0p5b \
  --dense-value-embeds \
  --value-embeds-layers 1,3,5,7,9,11,13,15,17,19,21,23,25,27 \
  --value-embedding-lr 0.2 \
  --value-embedding-adam-beta1 0.8 \
  --value-embedding-adam-beta2 0.995 \
  --value-embedding-adam-eps 1e-10 \
  --value-embedding-weight-decay 0.001
