# Mixture-of-Memory Embedding

[🤗 Pretrained checkpoints](https://huggingface.co/jojo23333/MOME-NanoChat-D24-100B) · [Paper](https://www.alphaxiv.org/abs/2609.15126) · [Citation](#citation)

Training and evaluation code for MoME, with Nanochat, Llama/MobileLLM and Qwen3 backbones.

## Install

With Python 3.10+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --extra gpu
source .venv/bin/activate
```

Use `--extra cpu` instead for CPU tests. Full training requires multiple GPUs.

## Training

Configs are shell scripts grouped by model family, with one file per setting:

- `configs/nanochat/`: D12 baselines, grouped-token MoME and bigram MoME.
- `configs/mobilellm/`: 125M and 350M Base, STEM, VEmbedding and MoME.
- `configs/qwen3/`: 0.6B Base, STEM, VEmbedding and MoME.

Inspect a command, then run it:

```bash
bash configs/mobilellm/350m_mome_a2_5.sh --plan
bash configs/mobilellm/350m_mome_a2_5.sh --data-cache /scratch/mome/data
```

To run all 350M settings sequentially, stopping on failure:

```bash
(
  set -e
  for config in configs/mobilellm/350m_*.sh; do
    bash "$config" --data-cache /scratch/mome/data
  done
)
```

Each script lists its training flags, seeds, GPU count and data size. It downloads only the needed shards and runs the listed seeds sequentially. Logs and evaluation reports go to `runs/<family>/<setting>/`; checkpoints and W&B are disabled. Use `--reuse-data /path/to/shards` for an existing corpus, `--output-dir /path/to/run` to change output storage, or `--seeds 42` to select a seed. `--nproc-per-node` and `--device-batch-size` can change the hardware layout while keeping the total token batch fixed.

See [training notes](docs/training.md) for budgets, data splits and optimizer settings. Reported scores are in `results/`, grouped by model family; wall-time ratios retain the paper's original measurements. The D24/100B checkpoints linked above are available separately; their training configs are not included here.

## Tests

```bash
NANOCHAT_FLASH_IMPL=sdpa TORCH_COMPILE_DISABLE=1 python -m pytest -q
```

## Citation

```bibtex
@misc{li2026mome,
  title         = {{MoME}: Mixture-of-Memory Embeddings for Context-Aware Sparse Lookup},
  author        = {Muchen Li and Leonid Sigal and Renjie Liao},
  year          = {2026},
  eprint        = {2609.15126},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.15126}
}
```

## License

Based on [nanochat](https://github.com/karpathy/nanochat), under the [MIT license](LICENSE). Datasets and benchmarks retain their respective licenses.
