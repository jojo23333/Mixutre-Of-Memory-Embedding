"""
Pretrain the base model.

Run from the repo root:

python -m scripts.base_train

or distributed:

torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- ...
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import statistics
import time
from contextlib import nullcontext

import torch
import wandb

from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    close_print0_log_file,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    print0,
    print_banner,
    set_print0_log_file,
)
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3, active_flash_impl, configured_flash_impl
from nanochat.gpt import GPT, GPTConfig
from nanochat.bigram_engram_gpt import BigramEngramGPT
from nanochat.loss_eval import evaluate_bpb
from nanochat.qwen3_0p5b_model import Qwen3_0p5B
from nanochat.report import get_report
from nanochat.slot_map_io import validate_slot_map_with_tokenizer
from nanochat.stem import compute_stem_hidden_dim, resolve_stem_layers
from nanochat.stemgpt_350m_model import StemGPT350M
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from nanochat.training_limits import elapsed_wall_time_seconds, should_stop_for_wall_time
from scripts.base_eval import evaluate_model


def _collect_moe_ve_balance_stats(model):
    loads = []
    maxvios = []
    eligible_slots = []
    value_embeds = getattr(model, "embed_value_moe", None)
    if value_embeds is not None and hasattr(value_embeds, "iter_layers"):
        for ve in value_embeds.iter_layers():
            load = getattr(ve, "last_load", None)
            if load is None or load.numel() == 0:
                continue
            loads.append(load.detach().float().cpu())
            window_count = getattr(ve, "maxvio_windows_completed", None)
            last_maxvio = getattr(ve, "last_maxvio", None)
            last_maxvio_slots = getattr(ve, "last_maxvio_eligible_slots", None)
            if (
                window_count is not None
                and int(window_count.item()) > 0
                and last_maxvio is not None
                and last_maxvio_slots is not None
                and int(last_maxvio_slots.item()) > 0
            ):
                maxvios.append(last_maxvio.detach().float().cpu())
                eligible_slots.append(last_maxvio_slots.detach().float().cpu())
    if not loads:
        return {}
    stds = torch.stack([l.std(unbiased=False) for l in loads])
    mins = torch.stack([l.min() for l in loads])
    maxs = torch.stack([l.max() for l in loads])
    stats = {
        "moe/load_std_mean": stds.mean().item(),
        "moe/load_min_mean": mins.mean().item(),
        "moe/load_max_mean": maxs.mean().item(),
        "moe/load_layers": len(loads),
    }
    if maxvios:
        stats["moe/maxvio_mean"] = torch.stack(maxvios).mean().item()
        stats["moe/maxvio_eligible_slots_mean"] = torch.stack(eligible_slots).mean().item()
        stats["moe/maxvio_layers"] = len(maxvios)
    return stats


def _evaluate_moe_ve_dataset_maxvio(model, eval_fn):
    value_embeds = getattr(model, "embed_value_moe", None)
    if value_embeds is None:
        return eval_fn(), None
    model.reset_moe_ve_eval_maxvio()
    try:
        result = eval_fn()
    finally:
        stats = model.finish_moe_ve_eval_maxvio()
    return result, stats


def _count_module_params(model):
    token_embed_params = 0
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        token_embed_params = model.transformer.wte.weight.numel()
    value_embedding_params = sum(
        p.numel() for p in model.parameters() if getattr(p, "value_embedding_is_table", False)
    )
    engram_embedding_params = sum(
        p.numel() for p in model.parameters() if getattr(p, "engram_is_embedding", False)
    )
    stem_embedding_params = sum(
        p.numel() for p in model.parameters() if getattr(p, "stem_is_embedding", False)
    )
    lm_head_params = 0
    if getattr(model, "lm_head", None) is not None:
        lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    other_params = total_params - token_embed_params - value_embedding_params - engram_embedding_params - stem_embedding_params - lm_head_params
    return {
        "token_embedding": token_embed_params,
        "value_embedding": value_embedding_params,
        "engram_embedding": engram_embedding_params,
        "stem_embedding": stem_embedding_params,
        "lm_head": lm_head_params,
        "other": other_params,
    }


def _parse_moe_setting(setting: str) -> tuple[bool, int, int]:
    if not setting:
        return True, 0, 0
    parts = setting.strip().split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid MoE setting '{setting}'. Expected format like '1_2_4'.")
    shared_flag = int(parts[0])
    activated = int(parts[1])
    experts = int(parts[2])
    if shared_flag not in (0, 1):
        raise ValueError("--moe-ve-setting shared flag must be 0 or 1")
    return bool(shared_flag), activated, experts


def _parse_layer_list(spec: str) -> list[int]:
    if not spec.strip():
        return []
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_seed_ids(spec: str, default_seed: int) -> list[int]:
    if not spec.strip():
        return [default_seed]
    seeds = []
    seen = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        seed = int(item)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    if not seeds:
        raise ValueError("--seed-ids must contain at least one integer seed")
    return seeds


def _resolve_model_class(model_type: str):
    if model_type == "gpt":
        return GPT
    if model_type == "bigram_engram_gpt":
        return BigramEngramGPT
    if model_type == "stemgpt_350m":
        return StemGPT350M
    if model_type in {"qwen3_0p5b", "qwen3_0p5b_stem"}:
        return Qwen3_0p5B
    raise ValueError(f"Unknown model type: {model_type}")


def _validate_layer_ids(layer_ids: list[int], num_layers: int, flag_name: str):
    for layer_id in layer_ids:
        if layer_id < 0 or layer_id >= num_layers:
            raise ValueError(f"Invalid layer id {layer_id} for {flag_name} with n_layer={num_layers}")


def _default_value_embedding_layers(num_layers: int) -> list[int]:
    parity = (num_layers - 1) % 2
    return [i for i in range(num_layers) if i % 2 == parity]


def _multiseed_parent_tag(args) -> str:
    if args.model_tag:
        return args.model_tag
    return f"d{args.depth}_multiseed"


def _clone_args_for_seed(args, seed: int, multi_seed: bool):
    cloned = argparse.Namespace(**vars(args))
    cloned.seed = seed
    if multi_seed:
        base_tag = args.model_tag if args.model_tag else f"d{args.depth}"
        cloned.model_tag = f"{base_tag}_seed{seed}"
    return cloned


def _format_metric(value, fmt: str) -> str:
    if value is None:
        return "n/a"
    return fmt.format(value)


def _format_mean_std(values: list[float], fmt: str) -> str:
    if not values:
        return "n/a"
    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{fmt.format(mean_value)} ± {fmt.format(std_value)}"


def _build_multiseed_tables(run_summaries):
    per_seed_lines = [
        "| seed | model_tag | train_bpb | val_bpb | core_metric | training_minutes | elapsed_minutes | peak_vram_mb | stop_reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in run_summaries:
        per_seed_lines.append(
            "| {seed} | {model_tag} | {train_bpb} | {val_bpb} | {core_metric} | {training_minutes} | {elapsed_minutes} | {peak_vram_mb} | {stop_reason} |".format(
                seed=run["seed"],
                model_tag=run["model_tag"],
                train_bpb=_format_metric(run["train_bpb"], "{:.6f}"),
                val_bpb=_format_metric(run["val_bpb"], "{:.6f}"),
                core_metric=_format_metric(run["core_metric"], "{:.4f}"),
                training_minutes=_format_metric(run["training_minutes"], "{:.2f}"),
                elapsed_minutes=_format_metric(run["elapsed_minutes"], "{:.2f}"),
                peak_vram_mb=_format_metric(run["peak_vram_mb"], "{:.2f}"),
                stop_reason=run["stop_reason"],
            )
        )

    aggregate_specs = [
        ("train_bpb", "{:.6f}"),
        ("val_bpb", "{:.6f}"),
        ("core_metric", "{:.4f}"),
        ("training_minutes", "{:.2f}"),
        ("elapsed_minutes", "{:.2f}"),
        ("peak_vram_mb", "{:.2f}"),
        ("total_flops", "{:.6e}"),
        ("num_params_M", "{:.1f}"),
    ]
    aggregate_lines = [
        "| metric | mean ± std |",
        "| --- | --- |",
    ]
    for key, fmt in aggregate_specs:
        values = [run[key] for run in run_summaries if run[key] is not None]
        aggregate_lines.append(f"| {key} | {_format_mean_std(values, fmt)} |")

    return "\n".join(per_seed_lines) + "\n", "\n".join(aggregate_lines) + "\n"


def _barrier_if_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Pretrain base model")
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="enable wandb logging; the displayed wandb run name follows model_tag",
    )
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument(
        "--model-type",
        type=str,
        default="gpt",
        choices=["gpt", "stemgpt_350m", "qwen3_0p5b", "qwen3_0p5b_stem", "bigram_engram_gpt"],
        help="base model architecture variant",
    )
    parser.add_argument("--seed", type=int, default=42, help="global training seed for weight init and dataloader state")
    parser.add_argument(
        "--seed-ids",
        type=str,
        default="",
        help="comma-separated seeds to run sequentially (default: use --seed once)",
    )
    parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
    parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
    parser.add_argument(
        "--model-dim",
        type=int,
        default=-1,
        help="explicit model dimension override (-1 = derive from depth * aspect-ratio)",
    )
    parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention; explicit for Qwen3 variants")
    parser.add_argument(
        "--num-heads",
        type=int,
        default=-1,
        help="explicit query-head count override (-1 = derive from model_dim / head_dim)",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=-1,
        help="explicit KV-head count override (-1 = same as query heads)",
    )
    parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
    parser.add_argument("--weight-tying", action="store_true", help="tie LM head weights to token embeddings")
    parser.add_argument(
        "--stem-layers",
        type=str,
        default="",
        help="comma-separated layer ids (0-based) that use STEM FFNs; use 'none' for the dense gated baseline (default = uniform one-third replacement)",
    )
    parser.add_argument(
        "--stem-embedding-dim",
        type=int,
        default=-1,
        help="explicit STEM embedding dim (-1 = derive from the gated FFN hidden dim)",
    )
    parser.add_argument(
        "--stem-multiple-of",
        type=int,
        default=256,
        help="round the gated/STEM FFN hidden dim up to a multiple of this value",
    )
    parser.add_argument(
        "--stem-ffn-dim-multiplier",
        type=float,
        default=-1.0,
        help="optional multiplier applied before rounding the gated/STEM FFN hidden dim (-1 = disable)",
    )
    parser.add_argument(
        "--window-pattern",
        type=str,
        default="L",
        help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')",
    )
    parser.add_argument("--moe-ve", action="store_true", help="enable MoE value embeddings")
    parser.add_argument(
        "--moe-ve-per-head-table",
        action="store_true",
        help="use separate routed value tables per KV head while keeping per-head routing",
    )
    parser.add_argument(
        "--moe-ve-network-shared-table",
        action="store_true",
        help="share MoE VE memory tables across value-embedding layers while keeping per-layer routers",
    )
    parser.add_argument("--dense-value-embeds", action="store_true", help="enable dense value embeddings (no routing)")
    parser.add_argument(
        "--value-embeds-layers",
        type=str,
        default="",
        help="comma-separated layer ids (0-based) for MoE value embeddings (default = alternating)",
    )
    parser.add_argument(
        "--bigram-engram-dict-size",
        type=int,
        default=6,
        help="dictionary size multiplier for bigram Engram hashing",
    )
    parser.add_argument(
        "--bigram-engram",
        action="store_true",
        help="enable shared bigram Engram injection on the selected backbone",
    )
    parser.add_argument(
        "--moe-ve-setting",
        type=str,
        default="",
        help="MoE value embedding setting as shared_activated_experts, e.g. 1_2_4",
    )
    parser.add_argument(
        "--moe-ve-gate-nl",
        type=str,
        default="sigmoid",
        help="MoE value routing mode: softmax|softmax-norm|softmax-norm-detach|sigmoid|sigmoid-norm",
    )
    parser.add_argument(
        "--moe-ve-gate-type",
        type=str,
        default="linear",
        help="MoE value router head type: linear|conv1d",
    )
    parser.add_argument(
        "--moe-ve-conv-kernel-size",
        type=int,
        default=4,
        help="kernel size for --moe-ve-gate-type=conv1d; uses causal left padding",
    )
    parser.add_argument(
        "--moe-ve-router-input",
        type=str,
        default="value",
        help="Router input for MoE value embeddings: value|hidden",
    )
    parser.add_argument(
        "--moe-ve-slot-mapping",
        type=str,
        default="none",
        help="token-to-slot mapping for MoE value tables: none|mod|headmod|table",
    )
    parser.add_argument(
        "--moe-ve-slot-index-mode",
        type=str,
        default="token",
        help="first-stage MoE VE slot index mode: token|bigram",
    )
    parser.add_argument(
        "--moe-ve-bigram-slot-factor",
        type=int,
        default=1,
        help="bigram MoE VE slot vocabulary multiplier N=factor*V; only used when slot index mode is bigram",
    )
    parser.add_argument(
        "--moe-ve-slot-factor",
        type=int,
        default=1,
        help="compression factor for slot sharing (slot_vocab_size = ceil(vocab_size / factor))",
    )
    parser.add_argument(
        "--moe-ve-slot-vocab-size",
        type=int,
        default=0,
        help="explicit slot vocabulary size for MoE value tables (0 = derive from mapping/factor)",
    )
    parser.add_argument(
        "--moe-ve-slot-dedicated-size",
        type=int,
        default=0,
        help="number of low-rank token ids that keep dedicated slots when mapping=headmod",
    )
    parser.add_argument(
        "--moe-ve-slot-map-path",
        type=str,
        default="",
        help="path to an explicit token_id->slot_id tensor when mapping=table",
    )
    parser.add_argument(
        "--moe-ve-balance-lr",
        type=float,
        default=0.001,
        help="slot-conditioned router bias update step for loss-free load balancing (0 disables)",
    )
    parser.add_argument(
        "--moe-ve-bias-scope",
        type=str,
        default="slot",
        help=(
            "router-bias balancing scope: 'slot' balances per routed token slot/head; "
            "'slot_shared_head' aggregates heads per slot/expert for shared tables"
        ),
    )
    parser.add_argument(
        "--moe-ve-bias-update",
        type=str,
        default="deepseek_moe",
        help="bias update rule for loss-free balancing: none|deepseek_moe|trust_region",
    )
    parser.add_argument(
        "--moe-ve-bias-min-visits",
        type=int,
        default=0,
        help="minimum per-slot visits before applying a trust-region bias update (0 = num_experts)",
    )
    parser.add_argument(
        "--moe-ve-bias-powerlaw-n",
        type=float,
        default=1.4,
        help="power-law base used to derive the relaxed trust-region share band (>= 1)",
    )
    parser.add_argument(
        "--moe-ve-maxvio-window",
        type=int,
        default=0,
        help="estimate slot-wise MaxVio over this many training steps (0 disables)",
    )
    parser.add_argument(
        "--moe-balance-mode",
        type=str,
        default="bias",
        help="MoE load balancing mode. Only 'bias' is supported in the active baseline.",
    )
    parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
    parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
    parser.add_argument(
        "--target-param-data-ratio",
        type=float,
        default=8,
        help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)",
    )
    parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
    parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
    parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
    parser.add_argument(
        "--stem-embedding-lr",
        type=float,
        default=-1.0,
        help="learning rate for STEM token embeddings (Adam, -1 = embedding_lr)",
    )
    parser.add_argument(
        "--value-embedding-lr",
        type=float,
        default=0.1,
        help="learning rate for value embedding parameters (Adam, -1 = embedding_lr)",
    )
    parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
    parser.add_argument("--weight-decay", type=float, default=0.2, help="weight decay for Muon weight matrices")
    parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
    parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
    parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
    parser.add_argument("--value-embedding-adam-beta1", type=float, default=-1.0, help="Adam beta1 for the VE table only (-1 = inherit --adam-beta1)")
    parser.add_argument("--value-embedding-adam-beta2", type=float, default=-1.0, help="Adam beta2 for the VE table only (-1 = inherit --adam-beta2)")
    parser.add_argument("--value-embedding-adam-eps", type=float, default=-1.0, help="Adam eps for the VE table only (-1 = default 1e-10)")
    parser.add_argument("--value-embedding-weight-decay", type=float, default=-1.0, help="Weight decay for the VE table only (-1 = default 0.0)")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
    parser.add_argument("--warmdown-ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
    parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
    parser.add_argument(
        "--disable-lr-decay",
        "--no-lr-decay",
        dest="disable_lr_decay",
        action="store_true",
        help="disable LR decay (keep LR constant after warmup)",
    )
    parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
    parser.add_argument(
        "--save-at-flops",
        type=str,
        default="",
        help="comma-separated training FLOPs at which to save checkpoints",
    )
    parser.add_argument(
        "--save-optim-state",
        action="store_true",
        help="save optimizer state in checkpoints (required for exact resume)",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="disable checkpoint saving entirely",
    )
    parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
    parser.add_argument(
        "--eval-every-flops",
        type=str,
        default="",
        help="FLOPs interval or comma-separated targets for full evaluation (overrides eval-every and core-metric-every)",
    )
    parser.add_argument("--eval-tokens", type=int, default=40 * 524288, help="number of tokens to evaluate val loss on")
    parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
    parser.add_argument("--core-metric-max-per-task", type=int, default=-1, help="examples per task for CORE metric")
    parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
    parser.add_argument(
        "--final-eval-splits",
        type=str,
        default="train,val",
        help="comma-separated splits to evaluate at the end: train,val",
    )
    parser.add_argument(
        "--skip-final-core",
        action="store_true",
        help="skip the final CORE evaluation to reduce overhead",
    )
    parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
    parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
    parser.add_argument(
        "--max-wall-time-seconds",
        type=float,
        default=-1.0,
        help="stop after this much end-to-end wall time from program start, then run final evaluation",
    )
    return parser


def _build_checkpoint_meta(
    *,
    step,
    val_bpb,
    model_type,
    model_config_kwargs,
    user_config,
    args,
    dataloader_state_dict,
    min_val_bpb,
    smooth_train_loss,
    smooth_train_aux_loss,
    total_training_time,
    stop_reason,
    elapsed_wall_time_seconds_value,
    extra=None,
):
    meta = {
        "step": step,
        "val_bpb": val_bpb,
        "model_type": model_type,
        "model_config": model_config_kwargs,
        "user_config": user_config,
        "device_batch_size": args.device_batch_size,
        "max_seq_len": args.max_seq_len,
        "dataloader_state_dict": dataloader_state_dict,
        "stop_reason": stop_reason,
        "elapsed_wall_time_seconds": elapsed_wall_time_seconds_value,
        "max_wall_time_seconds": args.max_wall_time_seconds,
        "loop_state": {
            "min_val_bpb": min_val_bpb,
            "smooth_train_loss": smooth_train_loss,
            "smooth_train_aux_loss": smooth_train_aux_loss,
            "total_training_time": total_training_time,
        },
    }
    if extra:
        meta.update(extra)
    return meta


def _run_single_seed(args, cleanup_distributed: bool = True):
    user_config = vars(args).copy()

    final_eval_splits = []
    if args.final_eval_splits.strip():
        for item in args.final_eval_splits.split(","):
            split = item.strip()
            if not split:
                continue
            if split not in {"train", "val"}:
                raise ValueError("--final-eval-splits entries must be chosen from: train,val")
            final_eval_splits.append(split)
    if not final_eval_splits:
        raise ValueError("--final-eval-splits must include at least one of: train,val")

    if args.moe_balance_mode != "bias":
        raise ValueError("Only --moe-balance-mode=bias is supported in the active baseline.")
    if args.moe_ve_bias_scope not in {"slot", "slot_shared_head"}:
        raise ValueError("--moe-ve-bias-scope must be one of: slot, slot_shared_head")
    if args.moe_ve_maxvio_window < 0:
        raise ValueError("--moe-ve-maxvio-window must be >= 0")

    script_start_time = time.time()
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, _, ddp_world_size, device = compute_init(device_type, seed=args.seed)
    master_process = ddp_rank == 0
    report = get_report(model_tag=args.model_tag)
    run_log_path = None
    if master_process and hasattr(report, "report_dir"):
        run_log_path = os.path.join(report.report_dir, "base-train.log")
        set_print0_log_file(run_log_path, mode="w")
    print_banner()
    if master_process and run_log_path:
        print0(f"Saving console logs to: {run_log_path}")
    print0(f"Training seed: {args.seed}")
    if args.skip_checkpoint:
        print0("Checkpoint saving disabled via --skip-checkpoint.")
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
    if device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gpu_device_name = torch.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_device_name)
        print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    else:
        gpu_peak_flops = float("inf")

    wandb_enabled = args.enable_wandb
    use_dummy_wandb = not wandb_enabled or not master_process
    wandb_run_name = args.model_tag if args.model_tag else f"d{args.depth}"
    wandb_run = (
        DummyWandb()
        if use_dummy_wandb
        else wandb.init(project="nanochat", name=wandb_run_name, config=user_config)
    )

    requested_flash_impl = configured_flash_impl()
    selected_flash_impl = active_flash_impl()
    if selected_flash_impl == "fa3":
        detail = "forced via NANOCHAT_FLASH_IMPL=fa3" if requested_flash_impl == "fa3" else "auto-selected on Hopper"
        print0(f"Attention backend: Flash Attention 3 ({detail}).")
    else:
        detail = "default on this stack" if requested_flash_impl == "sdpa" else "FA3 unavailable for auto selection"
        print0(f"Attention backend: PyTorch SDPA ({detail}).")
        if HAS_FA3 and requested_flash_impl == "sdpa":
            print0(
                "FA3 is available on this Hopper GPU but remains opt-in because longer training on this stack "
                "showed step-over-step memory growth and eventual OOM."
            )

    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    model_type = args.model_type
    num_layers = args.depth
    base_dim = args.depth * args.aspect_ratio
    if args.num_heads > 0 and args.model_dim <= 0:
        raise ValueError("--num-heads requires --model-dim so the attention geometry is explicit")
    if args.model_dim > 0:
        model_dim = args.model_dim
        if args.num_heads <= 0 and model_dim % args.head_dim != 0:
            raise ValueError(f"--model-dim ({model_dim}) must be divisible by --head-dim ({args.head_dim})")
    else:
        model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = args.num_heads if args.num_heads > 0 else model_dim // args.head_dim
    if num_heads <= 0 or model_dim % num_heads != 0:
        raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})")
    num_kv_heads = num_heads if args.num_kv_heads <= 0 else args.num_kv_heads
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    head_dim = args.head_dim if model_type in {"qwen3_0p5b", "qwen3_0p5b_stem"} else model_dim // num_heads
    if head_dim <= 0:
        raise ValueError("--head-dim must be > 0")
    print0(f"num_layers: {num_layers}")
    if args.model_dim > 0:
        print0(f"model_dim: {model_dim} (explicit override; depth*aspect_ratio base={base_dim})")
    else:
        print0(f"model_dim: {model_dim} (base: {base_dim}, nudge: {model_dim - base_dim:+d})")
    print0(f"num_heads: {num_heads}")
    print0(f"head_dim: {head_dim}")
    print0(f"num_kv_heads: {num_kv_heads}")
    print0(f"model_type: {args.model_type}")

    if args.stem_multiple_of <= 0:
        raise ValueError("--stem-multiple-of must be > 0")
    stem_layers = None
    stem_embedding_dim = None
    stem_ffn_dim_multiplier = None if args.stem_ffn_dim_multiplier <= 0 else args.stem_ffn_dim_multiplier
    if model_type in {"stemgpt_350m", "qwen3_0p5b_stem"}:
        stem_layers_spec = args.stem_layers.strip()
        if stem_layers_spec.lower() == "none":
            stem_layers = []
        elif stem_layers_spec:
            stem_layers = resolve_stem_layers(num_layers, _parse_layer_list(stem_layers_spec))
        else:
            stem_layers = (
                list(range(1, num_layers, 2))
                if model_type == "qwen3_0p5b_stem"
                else None if model_type == "stemgpt_350m"
                else resolve_stem_layers(num_layers, None)
            )
        if model_type == "qwen3_0p5b_stem":
            expected_stem_dim = 3 * model_dim
        else:
            expected_stem_dim = compute_stem_hidden_dim(
                dim=model_dim,
                multiple_of=args.stem_multiple_of,
                ffn_dim_multiplier=stem_ffn_dim_multiplier,
            )
        stem_embedding_dim = expected_stem_dim if args.stem_embedding_dim <= 0 else args.stem_embedding_dim
        if stem_embedding_dim != expected_stem_dim:
            raise ValueError(
                f"--stem-embedding-dim ({stem_embedding_dim}) must match resolved gated FFN dim ({expected_stem_dim})"
            )
        print0(f"STEM layers: {stem_layers}")
        print0(f"STEM embedding dim: {stem_embedding_dim}")
    else:
        if args.stem_layers.strip():
            print0(f"Ignoring --stem-layers because model_type={model_type}")
        if args.stem_embedding_dim > 0:
            print0(f"Ignoring --stem-embedding-dim because model_type={model_type}")

    dense_ve_enabled = args.dense_value_embeds
    moe_ve_enabled = args.moe_ve
    bigram_engram_enabled = args.bigram_engram or model_type == "bigram_engram_gpt"
    if args.bigram_engram and model_type not in {"gpt", "stemgpt_350m", "bigram_engram_gpt"}:
        raise ValueError("--bigram-engram is currently supported for model_type=gpt or stemgpt_350m")
    if args.moe_ve_per_head_table and not moe_ve_enabled:
        raise ValueError("--moe-ve-per-head-table requires --moe-ve")
    if args.moe_ve_network_shared_table and not moe_ve_enabled:
        raise ValueError("--moe-ve-network-shared-table requires --moe-ve")
    if dense_ve_enabled and moe_ve_enabled:
        raise ValueError("--dense-value-embeds and --moe-ve cannot both be enabled")
    if bigram_engram_enabled and (dense_ve_enabled or moe_ve_enabled):
        raise ValueError("bigram Engram does not support --dense-value-embeds or --moe-ve")
    if args.bigram_engram_dict_size < 1:
        raise ValueError("--bigram-engram-dict-size must be >= 1")
    value_embeds_layers = sorted(set(_parse_layer_list(args.value_embeds_layers)))
    _validate_layer_ids(value_embeds_layers, num_layers, "--value-embeds-layers")
    if (dense_ve_enabled or moe_ve_enabled) and not value_embeds_layers:
        value_embeds_layers = _default_value_embedding_layers(num_layers)
    if value_embeds_layers and not (dense_ve_enabled or moe_ve_enabled):
        print0("value_embeds_layers provided but no value-embedding mode is enabled; ignoring the layer list.")
        value_embeds_layers = []
    bigram_engram_layers = _default_value_embedding_layers(num_layers) if bigram_engram_enabled else []
    if bigram_engram_layers:
        print0(f"Bigram Engram layers: {bigram_engram_layers}")

    moe_ve_shared, moe_ve_num_activated, moe_ve_num_experts = _parse_moe_setting(args.moe_ve_setting)
    valid_gate_nl = {"softmax", "softmax-norm", "softmax-norm-detach", "sigmoid", "sigmoid-norm"}
    if args.moe_ve_gate_nl not in valid_gate_nl:
        raise ValueError(
            "--moe-ve-gate-nl must be one of: softmax, softmax-norm, softmax-norm-detach, sigmoid, sigmoid-norm"
        )
    if args.moe_ve_gate_type not in {"linear", "conv1d"}:
        raise ValueError("--moe-ve-gate-type must be one of: linear, conv1d")
    if args.moe_ve_conv_kernel_size < 1:
        raise ValueError("--moe-ve-conv-kernel-size must be >= 1")
    if args.moe_ve_router_input not in {"value", "hidden"}:
        raise ValueError("--moe-ve-router-input must be one of: value, hidden")
    if args.moe_ve_slot_mapping not in {"none", "mod", "headmod", "table"}:
        raise ValueError("--moe-ve-slot-mapping must be one of: none, mod, headmod, table")
    if args.moe_ve_slot_index_mode not in {"token", "bigram"}:
        raise ValueError("--moe-ve-slot-index-mode must be one of: token, bigram")
    if args.moe_ve_bigram_slot_factor < 1:
        raise ValueError("--moe-ve-bigram-slot-factor must be >= 1")
    if args.moe_ve_bias_update not in {"none", "deepseek_moe", "trust_region"}:
        raise ValueError("--moe-ve-bias-update must be one of: none, deepseek_moe, trust_region")
    if args.moe_ve_slot_factor < 1:
        raise ValueError("--moe-ve-slot-factor must be >= 1")
    if args.moe_ve_bias_min_visits < 0:
        raise ValueError("--moe-ve-bias-min-visits must be >= 0")
    if args.moe_ve_bias_powerlaw_n < 1.0:
        raise ValueError("--moe-ve-bias-powerlaw-n must be >= 1.0")
    if moe_ve_enabled:
        if moe_ve_num_experts < 0 or moe_ve_num_activated < 0:
            raise ValueError("--moe-ve-setting must define non-negative expert counts")
        if moe_ve_num_experts == 0 and not moe_ve_shared:
            raise ValueError("moe_ve requires a shared expert or routed experts")
        if moe_ve_num_experts == 0 and moe_ve_num_activated > 0:
            raise ValueError("--moe-ve-setting requires experts > 0 when activated > 0")
        if moe_ve_num_activated > moe_ve_num_experts:
            raise ValueError("--moe-ve-setting activated cannot exceed experts")
        if args.moe_ve_gate_nl == "softmax-norm" and moe_ve_num_activated == 1:
            print0(
                "Warning: --moe-ve-gate-nl softmax-norm with k=1 makes the kept routing weight constant at 1."
            )
        if args.moe_ve_bias_update == "trust_region" and args.moe_ve_bias_scope not in {"slot", "slot_shared_head"}:
            raise ValueError(
                "--moe-ve-bias-update=trust_region requires --moe-ve-bias-scope=slot or slot_shared_head"
            )
        if args.moe_ve_slot_index_mode == "bigram" and model_type != "gpt":
            raise ValueError("--moe-ve-slot-index-mode=bigram is currently supported only for --model-type=gpt")
        if args.moe_ve_slot_index_mode == "bigram" and args.moe_ve_slot_mapping != "none":
            raise ValueError("--moe-ve-slot-index-mode=bigram requires --moe-ve-slot-mapping=none")

    if args.moe_ve_slot_index_mode == "bigram":
        if args.moe_ve_slot_factor != 1:
            raise ValueError("--moe-ve-slot-factor must be 1 when --moe-ve-slot-index-mode=bigram")
        if args.moe_ve_slot_vocab_size != 0:
            raise ValueError("--moe-ve-slot-vocab-size must be 0 when --moe-ve-slot-index-mode=bigram")
        moe_ve_slot_vocab_size = vocab_size * args.moe_ve_bigram_slot_factor
    elif args.moe_ve_slot_mapping == "none":
        if args.moe_ve_slot_factor != 1:
            raise ValueError("--moe-ve-slot-factor must be 1 when --moe-ve-slot-mapping=none")
        if args.moe_ve_slot_vocab_size not in (0, vocab_size):
            raise ValueError("--moe-ve-slot-vocab-size must be 0 or vocab_size when mapping=none")
        moe_ve_slot_vocab_size = vocab_size
    else:
        if args.moe_ve_slot_vocab_size > 0:
            moe_ve_slot_vocab_size = args.moe_ve_slot_vocab_size
        else:
            moe_ve_slot_vocab_size = (vocab_size + args.moe_ve_slot_factor - 1) // args.moe_ve_slot_factor
        if moe_ve_slot_vocab_size <= 0 or moe_ve_slot_vocab_size > vocab_size:
            raise ValueError("--moe-ve-slot-vocab-size must be in [1, vocab_size]")

    # Pad slot_vocab_size so (slot_vocab_size * num_experts) is divisible by world_size,
    # required by the distributed AdamW optimizer for reduce_scatter sharding.
    if moe_ve_num_experts > 0 and ddp_world_size > 1:
        from math import gcd
        alignment = ddp_world_size // gcd(moe_ve_num_experts, ddp_world_size)
        old_svs = moe_ve_slot_vocab_size
        moe_ve_slot_vocab_size = ((moe_ve_slot_vocab_size + alignment - 1) // alignment) * alignment
        if moe_ve_slot_vocab_size != old_svs:
            print0(f"Padded slot_vocab_size {old_svs} -> {moe_ve_slot_vocab_size} for optimizer sharding (world_size={ddp_world_size})")

    if args.moe_ve_slot_mapping == "headmod":
        if args.moe_ve_slot_dedicated_size <= 0:
            raise ValueError("--moe-ve-slot-dedicated-size must be > 0 when mapping=headmod")
        if args.moe_ve_slot_dedicated_size >= moe_ve_slot_vocab_size:
            raise ValueError("--moe-ve-slot-dedicated-size must be smaller than slot_vocab_size when mapping=headmod")
    elif args.moe_ve_slot_dedicated_size != 0:
        raise ValueError("--moe-ve-slot-dedicated-size must be 0 unless mapping=headmod")

    if args.moe_ve_slot_mapping == "table":
        if not args.moe_ve_slot_map_path:
            raise ValueError("--moe-ve-slot-map-path must be provided when mapping=table")
        _, slot_map_meta = validate_slot_map_with_tokenizer(
            args.moe_ve_slot_map_path,
            tokenizer=tokenizer,
            expected_vocab_size=vocab_size,
            runtime_slot_vocab_size=moe_ve_slot_vocab_size,
            expected_slot_factor=args.moe_ve_slot_factor,
        )
        print0(
            "Validated table slot map: "
            f"format={slot_map_meta['format']} "
            f"file_slot_vocab_size={slot_map_meta.get('slot_vocab_size')} "
            f"runtime_slot_vocab_size={moe_ve_slot_vocab_size} "
            f"slot_size_histogram={slot_map_meta.get('slot_size_histogram', {})}"
        )
        slot_stats = slot_map_meta.get("stats", {})
        if slot_stats:
            print0(
                "Slot-map stats: "
                f"seen_tokens={slot_stats.get('seen_tokens', 'n/a')} "
                f"seen_docs={slot_stats.get('seen_docs', 'n/a')}"
            )
    elif args.moe_ve_slot_map_path:
        raise ValueError("--moe-ve-slot-map-path must be empty unless mapping=table")

    print0(f"MoE slot mapping: {args.moe_ve_slot_mapping}")
    print0(f"MoE network-shared table: {args.moe_ve_network_shared_table}")
    print0(f"MoE slot index mode: {args.moe_ve_slot_index_mode}")
    print0(f"MoE bigram slot factor: {args.moe_ve_bigram_slot_factor}")
    print0(f"MoE slot factor: {args.moe_ve_slot_factor}")
    print0(f"MoE slot vocab size: {moe_ve_slot_vocab_size}")
    print0(f"MoE dedicated head size: {args.moe_ve_slot_dedicated_size}")
    print0(f"MoE slot map path: {args.moe_ve_slot_map_path}")

    tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    assert args.total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
    print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
    print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
    print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

    batch_lr_scale = 1.0
    reference_batch_size = 2**19
    batch_ratio = args.total_batch_size / reference_batch_size
    if batch_ratio != 1.0:
        batch_lr_scale = batch_ratio ** 0.5
        print0(
            f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,} "
            f"(reference: {reference_batch_size:,})"
        )

    weight_decay_scaled = args.weight_decay * (12 / args.depth) ** 2
    if args.depth != 12:
        print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

    model_config_kwargs = dict(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
        head_dim=head_dim,
        weight_tying=args.weight_tying,
        window_pattern=args.window_pattern,
        stem_layers=stem_layers,
        stem_embedding_dim=stem_embedding_dim,
        stem_multiple_of=args.stem_multiple_of,
        stem_ffn_dim_multiplier=stem_ffn_dim_multiplier,
        value_embeds_layers=value_embeds_layers,
        dense_ve_enabled=dense_ve_enabled,
        moe_ve_enabled=moe_ve_enabled,
        moe_ve_per_head_table=args.moe_ve_per_head_table,
        moe_ve_network_shared_table=args.moe_ve_network_shared_table,
        moe_ve_setting=f"{int(moe_ve_shared)}_{moe_ve_num_activated}_{moe_ve_num_experts}",
        moe_ve_gate_nl=args.moe_ve_gate_nl,
        moe_ve_gate_type=args.moe_ve_gate_type,
        moe_ve_conv_kernel_size=args.moe_ve_conv_kernel_size,
        moe_ve_router_input=args.moe_ve_router_input,
        moe_ve_slot_mapping=args.moe_ve_slot_mapping,
        moe_ve_slot_index_mode=args.moe_ve_slot_index_mode,
        moe_ve_bigram_slot_factor=args.moe_ve_bigram_slot_factor,
        moe_ve_slot_vocab_size=moe_ve_slot_vocab_size,
        moe_ve_slot_dedicated_size=args.moe_ve_slot_dedicated_size,
        moe_ve_slot_map_path=args.moe_ve_slot_map_path,
        moe_ve_balance_lr=args.moe_ve_balance_lr,
        moe_balance_mode=args.moe_balance_mode,
        moe_ve_bias_scope=args.moe_ve_bias_scope,
        moe_ve_bias_update=args.moe_ve_bias_update,
        moe_ve_bias_min_visits=args.moe_ve_bias_min_visits,
        moe_ve_bias_powerlaw_n=args.moe_ve_bias_powerlaw_n,
        moe_ve_maxvio_window=args.moe_ve_maxvio_window,
    )
    if bigram_engram_enabled:
        model_config_kwargs.update(
            bigram_engram_enabled=True,
            bigram_engram_layers=bigram_engram_layers,
            bigram_engram_vocab_factor=args.bigram_engram_dict_size,
            bigram_engram_gate_channels=32,
            bigram_engram_init_lambda=0.1,
        )
    with torch.device("meta"):
        model_config = GPTConfig(**model_config_kwargs)
    ModelClass = _resolve_model_class(model_type)
    model = ModelClass(model_config)
    model.to_empty(device=device)
    model.init_weights()

    base_dir = get_base_dir()
    output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
    resuming = args.resume_from_step != -1
    optimizer_data = None
    meta_data = None
    if resuming:
        print0(f"Resuming optimization from step {args.resume_from_step}")
        model_data, optimizer_data, meta_data = load_checkpoint(
            checkpoint_dir,
            args.resume_from_step,
            device,
            load_optimizer=args.save_optim_state,
            rank=ddp_rank,
        )
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data

    orig_model = model
    model = torch.compile(model, dynamic=False)
    num_params = sum(p.numel() for p in model.parameters())
    num_scaling_params = orig_model.num_scaling_params()
    print0(f"Number of parameters: {num_params:,} (scaling: {num_scaling_params:,})")
    module_params = _count_module_params(orig_model)
    print0("Parameter breakdown:")
    print0(f"  token_embedding: {module_params['token_embedding']:,}")
    print0(f"  value_embedding: {module_params['value_embedding']:,}")
    print0(f"  engram_embedding: {module_params['engram_embedding']:,}")
    print0(f"  stem_embedding: {module_params['stem_embedding']:,}")
    print0(f"  lm_head: {module_params['lm_head']:,}")
    print0(f"  other: {module_params['other']:,}")

    num_flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
    if args.num_iterations > 0:
        num_iterations = args.num_iterations
        print0(f"Using user-provided number of iterations: {num_iterations:,}")
    elif args.target_flops > 0:
        num_iterations = round(args.target_flops / (num_flops_per_token * args.total_batch_size))
        print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
    elif args.target_param_data_ratio > 0:
        target_tokens = int(args.target_param_data_ratio * num_scaling_params)
        num_iterations = target_tokens // args.total_batch_size
        print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
    else:
        raise ValueError("No training horizon specified")
    total_tokens = args.total_batch_size * num_iterations
    print0(f"Total number of training tokens: {total_tokens:,}")
    print0(f"Tokens : Params ratio: {total_tokens / num_scaling_params:.2f}")
    total_training_flops = num_flops_per_token * total_tokens
    print0(f"Total training FLOPs estimate: {total_training_flops:e}")

    save_flops_targets = []
    if args.save_at_flops:
        for item in args.save_at_flops.split(","):
            item = item.strip()
            if item:
                save_flops_targets.append(float(item.replace("_", "")))
        save_flops_targets = sorted(f for f in save_flops_targets if f > 0)
    save_flops_index = 0

    eval_flops_targets = []
    if args.eval_every_flops:
        spec = args.eval_every_flops.strip()
        if "," in spec:
            for item in spec.split(","):
                item = item.strip()
                if item:
                    eval_flops_targets.append(float(item.replace("_", "")))
        else:
            interval = float(spec.replace("_", ""))
            if interval > 0:
                k = 1
                while interval * k <= total_training_flops:
                    eval_flops_targets.append(interval * k)
                    k += 1
        eval_flops_targets = sorted(f for f in eval_flops_targets if f > 0)
    eval_flops_index = 0

    adam_betas = (args.adam_beta1, args.adam_beta2)
    value_embedding_lr = args.embedding_lr if args.value_embedding_lr < 0 else args.value_embedding_lr
    stem_embedding_lr = args.embedding_lr if args.stem_embedding_lr < 0 else args.stem_embedding_lr
    engram_embedding_lr = 0.2 if bigram_engram_enabled else args.embedding_lr
    ve_b1 = args.adam_beta1 if args.value_embedding_adam_beta1 < 0 else args.value_embedding_adam_beta1
    ve_b2 = args.adam_beta2 if args.value_embedding_adam_beta2 < 0 else args.value_embedding_adam_beta2
    ve_betas = (ve_b1, ve_b2) if (args.value_embedding_adam_beta1 >= 0 or args.value_embedding_adam_beta2 >= 0) else None
    ve_eps = None if args.value_embedding_adam_eps < 0 else args.value_embedding_adam_eps
    ve_wd = None if args.value_embedding_weight_decay < 0 else args.value_embedding_weight_decay
    engram_betas = (0.8, 0.995) if bigram_engram_enabled else None
    engram_wd = None
    optimizers = model.setup_optimizers(
        unembedding_lr=args.unembedding_lr * batch_lr_scale,
        embedding_lr=args.embedding_lr * batch_lr_scale,
        matrix_lr=args.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
        adam_betas=adam_betas,
        scalar_lr=args.scalar_lr * batch_lr_scale,
        value_embedding_lr=value_embedding_lr * batch_lr_scale,
        engram_embedding_lr=engram_embedding_lr * batch_lr_scale,
        stem_embedding_lr=stem_embedding_lr * batch_lr_scale,
        value_embedding_adam_betas=ve_betas,
        value_embedding_adam_eps=ve_eps,
        value_embedding_weight_decay=ve_wd,
        engram_embedding_adam_betas=engram_betas,
        engram_embedding_weight_decay=engram_wd,
    )
    muon_optimizer = optimizers[1]
    if resuming and optimizer_data is not None:
        for opt, dat in zip(optimizers, optimizer_data):
            opt.load_state_dict(dat)
        del optimizer_data
    elif resuming and args.save_optim_state:
        print0("WARNING: --save-optim-state set but no optimizer state found; using fresh optimizer state.")
    elif resuming:
        print0("Resuming without optimizer state; using fresh optimizer state.")

    dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        split="train",
        device=device,
        resume_state_dict=dataloader_resume_state_dict,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        split="val",
        device=device,
    )
    x, y, dataloader_state_dict = next(train_loader)

    def get_lr_multiplier(it):
        warmup_iters = round(args.warmup_ratio * num_iterations)
        if it < warmup_iters:
            return 1.0 if warmup_iters == 0 else (it + 1) / warmup_iters
        if args.disable_lr_decay:
            return 1.0
        warmdown_iters = round(args.warmdown_ratio * num_iterations)
        if it <= num_iterations - warmdown_iters:
            return 1.0
        if warmdown_iters == 0:
            return args.final_lr_frac
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

    def get_muon_momentum(it):
        frac = min(it / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def get_weight_decay(it):
        return weight_decay_scaled * (1 - it / num_iterations)

    if not resuming:
        step = 0
        val_bpb = None
        min_val_bpb = float("inf")
        smooth_train_loss = 0.0
        smooth_train_aux_loss = 0.0
        total_training_time = 0.0
    else:
        step = meta_data["step"]
        loop_state = meta_data["loop_state"]
        val_bpb = meta_data["val_bpb"]
        min_val_bpb = loop_state["min_val_bpb"]
        smooth_train_loss = loop_state["smooth_train_loss"]
        smooth_train_aux_loss = loop_state.get("smooth_train_aux_loss", 0.0)
        total_training_time = loop_state["total_training_time"]

    save_flops_done = save_flops_index >= len(save_flops_targets)
    eval_flops_done = eval_flops_index >= len(eval_flops_targets)
    eval_flops_log = []
    results = {"core_metric": None, "centered_results": None}
    stop_reason = "completed"
    last_mfu = 0.0

    def run_full_eval(step_value, flops_so_far, reached_flops=None):
        nonlocal val_bpb, min_val_bpb, results
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        eval_steps = max(1, eval_steps)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step_value:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step_value:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step_value,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        if reached_flops:
            eval_flops_log.append(
                {
                    "step": step_value,
                    "flops": flops_so_far,
                    "targets": reached_flops,
                    "val_bpb": val_bpb,
                    "core_metric": results["core_metric"],
                }
            )
        model.train()

    while True:
        wall_limit_reached = should_stop_for_wall_time(script_start_time, args.max_wall_time_seconds)
        completed_training = step == num_iterations
        last_step = completed_training or wall_limit_reached
        if wall_limit_reached and not completed_training:
            stop_reason = "wall_time_limit"
            print0(
                "Wall-time limit reached after "
                f"{elapsed_wall_time_seconds(script_start_time):.1f}s; running final evaluation and exiting."
            )
        else:
            stop_reason = "completed"

        flops_so_far = num_flops_per_token * args.total_batch_size * step

        if (
            not args.skip_checkpoint
            and not save_flops_done
            and save_flops_index < len(save_flops_targets)
            and flops_so_far >= save_flops_targets[save_flops_index]
        ):
            reached = []
            while save_flops_index < len(save_flops_targets) and flops_so_far >= save_flops_targets[save_flops_index]:
                reached.append(save_flops_targets[save_flops_index])
                save_flops_index += 1
            optim_state = [opt.state_dict() for opt in optimizers] if args.save_optim_state else None
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                optim_state,
                _build_checkpoint_meta(
                    step=step,
                    val_bpb=val_bpb,
                    model_type=args.model_type,
                    model_config_kwargs=model_config_kwargs,
                    user_config=user_config,
                    args=args,
                    dataloader_state_dict=dataloader_state_dict,
                    min_val_bpb=min_val_bpb,
                    smooth_train_loss=smooth_train_loss,
                    smooth_train_aux_loss=smooth_train_aux_loss,
                    total_training_time=total_training_time,
                    stop_reason="in_progress",
                    elapsed_wall_time_seconds_value=elapsed_wall_time_seconds(script_start_time),
                    extra={"save_at_flops": reached},
                ),
                rank=ddp_rank,
            )
            print0(f"Saved checkpoint at step {step} for FLOPs targets: {', '.join(f'{x:e}' for x in reached)}")
            save_flops_done = save_flops_index >= len(save_flops_targets)

        if (
            eval_flops_targets
            and not eval_flops_done
            and eval_flops_index < len(eval_flops_targets)
            and flops_so_far >= eval_flops_targets[eval_flops_index]
        ):
            reached = []
            while eval_flops_index < len(eval_flops_targets) and flops_so_far >= eval_flops_targets[eval_flops_index]:
                reached.append(eval_flops_targets[eval_flops_index])
                eval_flops_index += 1
            run_full_eval(step, flops_so_far, reached_flops=reached)
            eval_flops_done = eval_flops_index >= len(eval_flops_targets)

        if not eval_flops_targets and args.eval_every > 0 and (last_step or step % args.eval_every == 0):
            model.eval()
            val_loader = build_val_loader()
            eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
            eval_steps = max(1, eval_steps)
            with autocast_ctx:
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
            wandb_run.log(
                {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "val/bpb": val_bpb,
                }
            )
            model.train()

        if not eval_flops_targets and args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
            model.eval()
            with autocast_ctx:
                results = evaluate_model(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
            print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
            wandb_run.log(
                {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "core_metric": results["core_metric"],
                    "centered_results": results["centered_results"],
                }
            )
            model.train()

        if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
            model.eval()
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(orig_model, tokenizer)
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                with autocast_ctx:
                    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0, seed=args.seed)
                print0(tokenizer.decode(sample[0]))
            model.train()

        if (
            not args.skip_checkpoint
            and (
                last_step
                or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0)
            )
        ):
            optim_state = [opt.state_dict() for opt in optimizers] if args.save_optim_state else None
            checkpoint_stop_reason = stop_reason if last_step else "in_progress"
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                optim_state,
                _build_checkpoint_meta(
                    step=step,
                    val_bpb=val_bpb,
                    model_type=args.model_type,
                    model_config_kwargs=model_config_kwargs,
                    user_config=user_config,
                    args=args,
                    dataloader_state_dict=dataloader_state_dict,
                    min_val_bpb=min_val_bpb,
                    smooth_train_loss=smooth_train_loss,
                    smooth_train_aux_loss=smooth_train_aux_loss,
                    total_training_time=total_training_time,
                    stop_reason=checkpoint_stop_reason,
                    elapsed_wall_time_seconds_value=elapsed_wall_time_seconds(script_start_time),
                ),
                rank=ddp_rank,
            )

        if last_step:
            break

        synchronize()
        t0 = time.time()
        for _ in range(grad_accum_steps):
            with autocast_ctx:
                loss, aux_loss = model(x, y, return_aux_loss=True)
            train_loss = loss.detach()
            train_aux_loss = aux_loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, dataloader_state_dict = next(train_loader)

        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(step)
        for group in muon_optimizer.param_groups:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        train_loss_f = train_loss.item()
        train_aux_loss_f = train_aux_loss.item()
        synchronize()
        t1 = time.time()
        dt = t1 - t0

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        smooth_train_aux_loss = ema_beta * smooth_train_aux_loss + (1 - ema_beta) * train_aux_loss_f
        debiased_smooth_aux_loss = smooth_train_aux_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * step / num_iterations
        tok_per_sec = int(args.total_batch_size / dt)
        flops_per_sec = num_flops_per_token * args.total_batch_size / dt
        last_mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
        if step > 10:
            total_training_time += dt
        steps_done = step - 10
        if steps_done > 0:
            avg_time_per_step = total_training_time / steps_done
            remaining_steps = num_iterations - step
            eta_seconds = remaining_steps * avg_time_per_step
            eta_str = f" | eta: {eta_seconds/60:.1f}m"
        else:
            eta_str = ""
        epoch = dataloader_state_dict["epoch"]
        elapsed_wall = elapsed_wall_time_seconds(script_start_time)
        print0(
            f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
            f"loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | "
            f"tok/sec: {tok_per_sec:,} | mfu: {last_mfu:.2f} | epoch: {epoch} | "
            f"total time: {total_training_time/60:.2f}m | wall: {elapsed_wall/60:.2f}m{eta_str}"
        )
        if step % 100 == 0:
            log_data = {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "elapsed_wall_time_seconds": elapsed_wall,
                "train/loss": debiased_smooth_loss,
                "train/aux_loss": debiased_smooth_aux_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": last_mfu,
                "train/epoch": epoch,
            }
            if master_process and moe_ve_enabled:
                moe_stats = _collect_moe_ve_balance_stats(orig_model)
                if moe_stats:
                    message = "MoE load | std_mean={std:.4f} min_mean={min:.4f} max_mean={max:.4f} layers={layers}".format(
                        std=moe_stats["moe/load_std_mean"],
                        min=moe_stats["moe/load_min_mean"],
                        max=moe_stats["moe/load_max_mean"],
                        layers=moe_stats["moe/load_layers"],
                    )
                    if "moe/maxvio_mean" in moe_stats:
                        message += " | maxvio_mean={maxvio:.4f} eligible_slots_mean={slots:.1f} maxvio_layers={layers}".format(
                            maxvio=moe_stats["moe/maxvio_mean"],
                            slots=moe_stats["moe/maxvio_eligible_slots_mean"],
                            layers=moe_stats["moe/maxvio_layers"],
                        )
                    print0(message)
                log_data.update(moe_stats)
            wandb_run.log(log_data)

        step += 1

    if master_process and moe_ve_enabled:
        moe_stats = _collect_moe_ve_balance_stats(orig_model)
        if moe_stats:
            message = "Final MoE load | std_mean={std:.4f} min_mean={min:.4f} max_mean={max:.4f} layers={layers}".format(
                std=moe_stats["moe/load_std_mean"],
                min=moe_stats["moe/load_min_mean"],
                max=moe_stats["moe/load_max_mean"],
                layers=moe_stats["moe/load_layers"],
            )
            if "moe/maxvio_mean" in moe_stats:
                message += " | maxvio_mean={maxvio:.4f} eligible_slots_mean={slots:.1f} maxvio_layers={layers}".format(
                    maxvio=moe_stats["moe/maxvio_mean"],
                    slots=moe_stats["moe/maxvio_eligible_slots_mean"],
                    layers=moe_stats["moe/maxvio_layers"],
                )
            print0(message)

    print0("\n" + "=" * 80)
    print0("Final Full Evaluation")
    print0("=" * 80)
    model.eval()
    eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
    eval_steps = max(1, eval_steps)
    final_bpb = {}
    final_moe_maxvio = {}
    final_moe_absolute_load_deviation = {}
    for split_name in final_eval_splits:
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer,
            args.device_batch_size,
            args.max_seq_len,
            split_name,
            device=device,
        )
        moe_eval_stats = None
        if args.moe_ve:
            def _run_split_eval():
                with autocast_ctx:
                    return evaluate_bpb(model, loader, eval_steps, token_bytes)

            bpb, moe_eval_stats = _evaluate_moe_ve_dataset_maxvio(orig_model, _run_split_eval)
        else:
            with autocast_ctx:
                bpb = evaluate_bpb(model, loader, eval_steps, token_bytes)
        final_bpb[split_name] = bpb
        print0(f"{split_name} bpb: {bpb:.6f}")
        if moe_eval_stats is not None:
            final_moe_maxvio[split_name] = moe_eval_stats["maxvio_mean"]
            final_moe_absolute_load_deviation[split_name] = moe_eval_stats["absolute_load_deviation_mean"]
            print0(
                f"{split_name} moe maxvio: {moe_eval_stats['maxvio_mean']:.4f} "
                f"(visited_slots_mean={moe_eval_stats['visited_slots_mean']:.1f}, layers={moe_eval_stats['layers']})"
            )
            print0(f"{split_name} moe absolute load deviation: {moe_eval_stats['absolute_load_deviation_mean']:.4f}")
    results = {"core_metric": None, "centered_results": None}
    if args.skip_final_core:
        print0("Skipping final CORE metric (--skip-final-core).")
    else:
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"CORE metric: {results['core_metric']:.4f}")
    model.train()
    val_bpb = final_bpb.get("val", val_bpb)
    if val_bpb is not None and val_bpb < min_val_bpb:
        min_val_bpb = val_bpb

    final_wall_time_seconds = elapsed_wall_time_seconds(script_start_time)
    if master_process:
        eval_report_data = [
            {
                "model": f"{output_dirname} (step {step})",
                "CORE metric": results["core_metric"],
                "train bpb": final_bpb.get("train"),
                "val bpb": final_bpb.get("val"),
                "stop reason": stop_reason,
                "elapsed wall time (s)": final_wall_time_seconds,
                "wall time limit (s)": args.max_wall_time_seconds,
            },
        ]
        if final_moe_maxvio:
            eval_report_data[0]["train moe maxvio"] = final_moe_maxvio.get("train")
            eval_report_data[0]["val moe maxvio"] = final_moe_maxvio.get("val")
            eval_report_data[0]["train moe absolute load deviation"] = final_moe_absolute_load_deviation.get("train")
            eval_report_data[0]["val moe absolute load deviation"] = final_moe_absolute_load_deviation.get("val")
        if results.get("centered_results") is not None:
            eval_report_data.append(results["centered_results"])
        report.log(section="Base model evaluation", data=eval_report_data)

    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time/60:.2f}m")
    print0(f"Elapsed wall time: {final_wall_time_seconds/60:.2f}m")
    print0(f"Stop reason: {stop_reason}")
    if args.max_wall_time_seconds > 0:
        print0(f"Wall-time limit: {args.max_wall_time_seconds:.1f}s")
    if val_bpb is not None:
        print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

    setup_stats = {
        "Number of parameters": num_params,
        "Params/token_embedding": module_params["token_embedding"],
        "Params/value_embedding": module_params["value_embedding"],
        "Params/stem_embedding": module_params["stem_embedding"],
        "Params/lm_head": module_params["lm_head"],
        "Params/other": module_params["other"],
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_tokens / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
        "stop_reason": stop_reason,
        "elapsed_wall_time_seconds": final_wall_time_seconds,
        "max_wall_time_seconds": args.max_wall_time_seconds,
    }
    if save_flops_targets:
        setup_stats["Save-at FLOPs targets"] = ",".join(f"{x:e}" for x in save_flops_targets)
    if eval_flops_targets:
        if args.eval_every_flops and "," not in args.eval_every_flops:
            setup_stats["Eval-at FLOPs interval"] = args.eval_every_flops
        setup_stats["Eval-at FLOPs targets"] = ",".join(f"{x:e}" for x in eval_flops_targets)

    eval_flops_report = None
    if eval_flops_log:
        lines = ["### Eval @ FLOPs"]
        for entry in eval_flops_log:
            targets = ", ".join(f"{t:e}" for t in entry["targets"])
            lines.append(
                "- step {step:05d} | flops {flops:.3e} | targets {targets} | val_bpb {val_bpb:.6f} | core {core_metric:.4f}".format(
                    step=entry["step"],
                    flops=entry["flops"],
                    targets=targets,
                    val_bpb=entry["val_bpb"],
                    core_metric=entry["core_metric"],
                )
            )
        eval_flops_report = "\n".join(lines) + "\n"

    report.log(
        section="Base model training",
        data=[
            user_config,
            setup_stats,
            eval_flops_report,
            {
                "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
                "Final validation bpb": val_bpb,
                "CORE metric estimate": results.get("core_metric", None),
                "MFU %": f"{last_mfu:.2f}%",
                "Total training flops": f"{num_flops_per_token * args.total_batch_size * step:e}",
                "Total training time": f"{total_training_time/60:.2f}m",
                "Elapsed wall time": f"{final_wall_time_seconds/60:.2f}m",
                "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
                "Stop reason": stop_reason,
            },
        ],
    )

    peak_vram_mb = get_max_memory() / 1024 / 1024
    elapsed_minutes = final_wall_time_seconds / 60
    print0("\n---")
    if final_bpb.get("train") is not None:
        print0(f"train_bpb:        {final_bpb['train']:.6f}")
    if val_bpb is not None:
        print0(f"val_bpb:          {val_bpb:.6f}")
    print0(f"training_minutes: {total_training_time/60:.2f}")
    print0(f"elapsed_minutes:  {elapsed_minutes:.2f}")
    print0(f"peak_vram_mb:     {peak_vram_mb:.2f}")
    print0(f"total_flops:      {num_flops_per_token * args.total_batch_size * step:e}")
    print0(f"num_params_M:     {num_params/1e6:.1f}")
    print0(f"stop_reason:      {stop_reason}")
    if results.get("core_metric") is not None:
        print0(f"core_metric:      {results['core_metric']:.4f}")

    run_summary = {
        "seed": args.seed,
        "model_tag": output_dirname,
        "train_bpb": final_bpb.get("train"),
        "val_bpb": val_bpb,
        "core_metric": results.get("core_metric"),
        "training_minutes": total_training_time / 60,
        "elapsed_minutes": elapsed_minutes,
        "peak_vram_mb": peak_vram_mb,
        "total_flops": num_flops_per_token * args.total_batch_size * step,
        "num_params_M": num_params / 1e6,
        "stop_reason": stop_reason,
    }

    wandb_run.finish()
    close_print0_log_file()
    if cleanup_distributed:
        compute_cleanup()
    return run_summary


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    seed_ids = _parse_seed_ids(args.seed_ids, default_seed=args.seed)
    multi_seed = len(seed_ids) > 1

    run_summaries = []
    for seed in seed_ids:
        run_args = _clone_args_for_seed(args, seed, multi_seed=multi_seed)
        run_summaries.append(_run_single_seed(run_args, cleanup_distributed=not multi_seed))
        if multi_seed:
            _barrier_if_distributed()

    if not multi_seed:
        return

    per_seed_table, aggregate_table = _build_multiseed_tables(run_summaries)
    summary_tag = _multiseed_parent_tag(args)
    summary_report = get_report(model_tag=summary_tag)
    if hasattr(summary_report, "reset"):
        summary_report.reset()
    summary_report.log(
        section="Base model multi-seed summary",
        data=[
            {
                "Parent model tag": summary_tag,
                "Seed ids": ",".join(str(seed) for seed in seed_ids),
                "Number of seeds": len(seed_ids),
                "Std definition": "sample std (n-1)",
            },
            "### Per-seed results\n" + per_seed_table,
            "### Aggregate results\n" + aggregate_table,
        ],
    )
    if hasattr(summary_report, "generate"):
        summary_report.generate()

    print0("\n" + "=" * 80)
    print0(f"Multi-seed summary | parent model tag: {summary_tag}")
    print0("=" * 80)
    print0(per_seed_table.rstrip())
    print0("")
    print0(aggregate_table.rstrip())
    compute_cleanup()


if __name__ == "__main__":
    main()
