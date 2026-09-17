import pytest

from nanochat.training_limits import elapsed_wall_time_seconds, should_stop_for_wall_time
from scripts.base_train import _parse_seed_ids, build_arg_parser


def test_elapsed_wall_time_never_negative():
    assert elapsed_wall_time_seconds(10.0, now=9.0) == 0.0


def test_should_stop_for_wall_time_when_limit_reached():
    assert not should_stop_for_wall_time(0.0, 10.0, now=9.9)
    assert should_stop_for_wall_time(0.0, 10.0, now=10.0)
    assert not should_stop_for_wall_time(0.0, -1.0, now=100.0)


def test_base_train_parser_accepts_wall_time_flag():
    parser = build_arg_parser()
    args = parser.parse_args(["--moe-ve", "--max-wall-time-seconds", "123.5"])
    assert args.moe_ve is True
    assert args.max_wall_time_seconds == 123.5


def test_base_train_parser_accepts_enable_wandb_flag():
    parser = build_arg_parser()
    args = parser.parse_args(["--moe-ve", "--enable-wandb"])
    assert args.moe_ve is True
    assert args.enable_wandb is True


def test_base_train_parser_accepts_stemgpt_350m_geometry_flags():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-type",
            "stemgpt_350m",
            "--model-dim",
            "576",
            "--num-heads",
            "9",
            "--num-kv-heads",
            "3",
            "--weight-tying",
        ]
    )
    assert args.model_type == "stemgpt_350m"
    assert args.model_dim == 576
    assert args.num_heads == 9
    assert args.num_kv_heads == 3
    assert args.weight_tying is True


def test_base_train_parser_accepts_qwen3_0p5b_model_type():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-type",
            "qwen3_0p5b",
            "--model-dim",
            "1024",
            "--num-heads",
            "16",
            "--head-dim",
            "128",
            "--num-kv-heads",
            "8",
            "--weight-tying",
        ]
    )
    assert args.model_type == "qwen3_0p5b"
    assert args.model_dim == 1024
    assert args.num_heads == 16
    assert args.head_dim == 128
    assert args.num_kv_heads == 8
    assert args.weight_tying is True


def test_base_train_parser_accepts_seed_list_and_dense_value_embeds():
    parser = build_arg_parser()
    args = parser.parse_args(["--dense-value-embeds", "--seed", "7", "--seed-ids", "41,42,43"])
    assert args.dense_value_embeds is True
    assert args.seed == 7
    assert args.seed_ids == "41,42,43"


def test_base_train_parser_accepts_moe_balance_scope_and_maxvio_flags():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--moe-ve",
        "--moe-ve-bias-scope", "global",
        "--moe-ve-maxvio-window", "10",
    ])
    assert args.moe_ve is True
    assert args.moe_ve_bias_scope == "global"
    assert args.moe_ve_maxvio_window == 10


def test_base_train_parser_accepts_trust_region_bias_flags():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--moe-ve",
        "--moe-ve-bias-update", "trust_region",
        "--moe-ve-bias-min-visits", "64",
        "--moe-ve-bias-powerlaw-n", "1.2",
    ])
    assert args.moe_ve_bias_update == "trust_region"
    assert args.moe_ve_bias_min_visits == 64
    assert args.moe_ve_bias_powerlaw_n == 1.2


def test_base_train_parser_accepts_none_bias_update():
    parser = build_arg_parser()
    args = parser.parse_args(["--moe-ve", "--moe-ve-bias-update", "none"])
    assert args.moe_ve_bias_update == "none"


def test_base_train_parser_rejects_removed_moe_bias_window_flag():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--moe-ve", "--moe-ve-bias-window"])


def test_base_train_parser_rejects_memory_reuse_flags():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--memory-refresh-layers", "1,3,5"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--memory-block-span", "2"])


def test_base_train_parser_accepts_moe_router_head_flags():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--moe-ve",
        "--moe-ve-gate-type", "conv1d",
        "--moe-ve-conv-kernel-size", "8",
    ])
    assert args.moe_ve is True
    assert args.moe_ve_gate_type == "conv1d"
    assert args.moe_ve_conv_kernel_size == 8


def test_base_train_parser_accepts_network_shared_moe_table_flag():
    parser = build_arg_parser()
    default_args = parser.parse_args(["--moe-ve"])
    args = parser.parse_args(["--moe-ve", "--moe-ve-network-shared-table"])

    assert default_args.moe_ve_network_shared_table is False
    assert args.moe_ve_network_shared_table is True


def test_base_train_parser_accepts_explicit_moe_gate_mode():
    parser = build_arg_parser()
    args = parser.parse_args(["--moe-ve", "--moe-ve-gate-nl", "softmax-norm-detach"])
    assert args.moe_ve is True
    assert args.moe_ve_gate_nl == "softmax-norm-detach"


def test_base_train_parser_rejects_removed_moe_gate_norm_flag():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--moe-ve", "--moe-ve-gate-norm"])


def test_base_train_parser_accepts_bigram_engram_model_type():
    parser = build_arg_parser()
    args = parser.parse_args(["--model-type", "bigram_engram_gpt"])
    assert args.model_type == "bigram_engram_gpt"
    assert args.bigram_engram_dict_size == 6


def test_parse_seed_ids_uses_explicit_list_and_deduplicates():
    assert _parse_seed_ids("", default_seed=42) == [42]
    assert _parse_seed_ids("41, 42,41,43", default_seed=99) == [41, 42, 43]
