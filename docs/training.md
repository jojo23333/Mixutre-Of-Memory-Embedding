# Training settings

The model and optimizer code is based on the NeurIPS version (`02f117e387de54cbccc9612ec0bd5149686a22f8`), with the Qwen-STEM extension. No historical checkout is needed to run these configs.

| Model / setting | Budget | Seeds | Training shards | GPUs | Attention |
|---|---|---|---|---|---|
| Nanochat D12: baselines and grouped-token MoME | 3e18 FLOPs | 42, 43, 44 | 100 | 4 | SDPA |
| Nanochat D12: bigram MoME | 3e18 FLOPs | 42, 43 | 100 | 4 | Auto |
| MobileLLM 125M | 9,536 steps | 42 | 160 | 4 | SDPA |
| MobileLLM 350M | 38,146 steps | 42 | 520 | 4 | SDPA |
| Qwen3 0.6B | 38,146 steps | 42 | 369 | 4; STEM: 8 | Auto |

All settings use sequences of 2,048 tokens and a total batch of 524,288 tokens. In filenames, `c` is the grouped-token compression factor, `a` is the number of active experts, `m` is the total expert count, and `n` is the bigram slot-vocabulary multiplier. For example, `350m_mome_a2_5.sh` uses two of five experts. The Bigram baseline is shared across the D12 comparisons and has one config, `d12_bigram.sh`.

The bigram MoME results average seeds 42 and 43, matching the archived runs; the manuscript describes three seeds. The shared Bigram baseline averages 42, 43 and 44.

## Optimizer and routing

Both MobileLLM sizes use value-embedding LR 0.1 and inherit Adam betas (0.8, 0.95). Nanochat and Qwen3 VEmbedding/MoME use LR 0.2 and betas (0.8, 0.995). Value-table weight decay is 0.001 and Adam epsilon is 1e-10 throughout. All MoME settings disable loss-free router bias updates and use no router-balancing auxiliary loss. The 125M configs retain an inactive balance-LR flag of 0.001.

## Data and evaluation

Training uses `karpathy/fineweb-edu-100b-shuffle`, taking the configured number of shards in order and skipping shard 369. Shard 369 is held out for validation. The runner shares a download cache across settings and records the exact corpus inventory for each run. The bundled tokenizer has 32,768 tokens; do not retrain it. The two grouped-token slot maps were produced by D24 but require no model checkpoint to use.

Qwen3 retains the original 369-training-shard subset. The Qwen-STEM log records approximately 15.565B tokens before cycling, so its 20B-token run includes a partial second pass.

Final evaluation reports train/validation BPB and CORE. The evaluator downloads the CORE benchmark bundle on first use. Automatic attention selects FA3 on supported Hopper GPUs and SDPA otherwise; the original Qwen-STEM run used FA3. Changing hardware, kernels or software versions can change scores and timing. The files in `results/` contain the paper's reported values, not new training outputs.
