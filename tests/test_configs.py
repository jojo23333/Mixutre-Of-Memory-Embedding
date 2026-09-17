import os
import shlex
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from scripts.run import ROOT, verify_assets
from scripts.base_train import build_arg_parser

CONFIGS = sorted((ROOT / 'configs').glob('*/*.sh'))


def plan(config, *flags):
    return subprocess.run(['bash', str(config), '--plan', '--run-id', 'test', *flags], cwd=ROOT, env=dict(os.environ, PYTHON=sys.executable), capture_output=True, text=True)


@pytest.mark.parametrize('config', CONFIGS, ids=lambda path: str(path.relative_to(ROOT / 'configs')))
def test_configs_parse_and_cover_recorded_seeds(config):
    subprocess.run(['bash', '-n', str(config)], check=True)
    result = plan(config)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    if config.parent.name == 'nanochat':
        expected_seeds = [42, 43] if '_mome_bigram_' in config.stem else [42, 43, 44]
    else:
        expected_seeds = [42]
    assert len(commands) == len(expected_seeds)
    for command, seed in zip(commands, expected_seeds):
        args = build_arg_parser().parse_args(command[command.index('--') + 1:])
        assert args.seed_ids == str(seed)
        assert args.skip_checkpoint
        assert not args.enable_wandb
        assert args.total_batch_size == 524288
        assert args.max_seq_len == 2048
        world = 8 if config.parent.name == 'qwen3' and config.stem.endswith('_stem') else 4
        assert f'--nproc_per_node={world}' in command
        if args.moe_ve:
            assert args.moe_ve_bias_update == 'none'
        if args.moe_ve or args.dense_value_embeds:
            assert args.value_embedding_weight_decay == 0.001
            if config.parent.name == 'mobilellm':
                assert args.value_embedding_lr == 0.1
                assert args.value_embedding_adam_beta2 == -1.0
                assert args.adam_beta2 == 0.95
            else:
                assert args.value_embedding_lr == 0.2
                assert args.value_embedding_adam_beta2 == 0.995


def test_config_inventory():
    assert len(CONFIGS) == 29
    assert not list((ROOT / 'configs').rglob('*.json'))
    assert not list(ROOT.rglob('table[0-9]*'))


@pytest.mark.parametrize('flags', [['--nproc-per-node', '0'], ['--seeds', 'not-a-seed'], ['--seeds', '42,42'], ['--seeds', '-1'], ['--seeds', ''], ['--run-id', '../bad'], ['--device-batch-size', '3'], ['--training-shards', '0']])
def test_invalid_plans_fail_without_training(flags):
    result = plan(ROOT / 'configs' / 'nanochat' / 'd12_base.sh', *flags)
    assert result.returncode != 0


def test_seed_and_hardware_overrides():
    result = plan(ROOT / 'configs' / 'nanochat' / 'd12_base.sh', '--seeds', '7', '--nproc-per-node', '2', '--device-batch-size', '16')
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    command = shlex.split(lines[0])
    args = build_arg_parser().parse_args(command[command.index('--') + 1:])
    assert args.seed_ids == '7'
    assert args.device_batch_size == 16
    assert args.total_batch_size == 524288
    assert '--nproc_per_node=2' in command


@pytest.mark.parametrize('status', [0, 1])
def test_runner_stages_data_and_stops_on_failure(tmp_path, monkeypatch, status):
    from scripts import run, prepare_data
    staged = []
    launched = []
    monkeypatch.setattr(run, 'verify_assets', lambda: None)
    monkeypatch.setattr(prepare_data, 'prepare_data', lambda *args: staged.append(args))
    class Process:
        stdout = ['training output\n']

        def __init__(self, command, **kwargs):
            launched.append((command, kwargs))

        def wait(self):
            return status

    monkeypatch.setattr(run.subprocess, 'Popen', Process)
    runtime = tmp_path / 'run'
    cache = tmp_path / 'cache'
    reuse = tmp_path / 'reuse'
    monkeypatch.setattr(sys, 'argv', ['run', '--name', 'nanochat/d12_base', '--training-shards', '100', '--seeds', '42,43', '--output-dir', str(runtime), '--data-cache', str(cache), '--reuse-data', str(reuse), '--run-id', 'test', '--', '--device-batch-size', '32', '--total-batch-size', '524288', '--max-seq-len', '2048', '--skip-checkpoint'])
    if status:
        with pytest.raises(SystemExit, match='Seed 42 failed'):
            run.main()
    else:
        run.main()
    assert staged == [(cache, runtime / 'base_data', 100, reuse)]
    assert len(launched) == (1 if status else 2)
    assert launched[0][1]['env']['NANOCHAT_BASE_DIR'] == str(runtime)
    assert launched[0][1]['env']['NANOCHAT_FLASH_IMPL'] == 'sdpa'
    assert (runtime / 'tokenizer' / 'tokenizer.pkl').is_file()
    logs = runtime / 'logs' / 'test'
    assert (logs / 'seed42.log').read_text() == 'training output\n'
    assert shlex.split((logs / 'seed42.command.txt').read_text()) == launched[0][0]
    with pytest.raises(FileExistsError):
        run.main()


def test_bundled_assets_and_tokenizer_slot_maps():
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.slot_map_io import validate_slot_map_with_tokenizer
    verify_assets()
    tokenizer = RustBPETokenizer.from_directory(str(ROOT / 'artifacts' / 'tokenizer'))
    assert tokenizer.get_vocab_size() == 32768
    for factor in (2, 4):
        path = next((ROOT / 'artifacts' / 'slotmaps').glob(f'*_k{factor}_*'))
        slot_map, metadata = validate_slot_map_with_tokenizer(path, tokenizer=tokenizer, expected_vocab_size=32768, runtime_slot_vocab_size=32768 // factor, expected_slot_factor=factor)
        assert slot_map.shape == (32768,)
        assert metadata['tokenizer_validated']


def test_data_staging_keeps_validation_out_of_larger_training_split(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests
    from scripts.prepare_data import prepare_data, training_indices
    from nanochat.dataset import list_parquet_files
    source = tmp_path / 'existing'
    source.mkdir()
    for index in [*training_indices(520), 369]:
        pq.write_table(pa.table({'text': [f'document {index}']}), source / f'shard_{index:05d}.parquet')
    monkeypatch.setattr(requests, 'get', lambda *a, **kw: pytest.fail('Reusable data must not be downloaded'))
    stage = tmp_path / 'stage'
    manifest = prepare_data(tmp_path / 'cache', stage, 520, source)
    files = list_parquet_files(stage)
    assert len(files) == 521
    assert files[-1].endswith('zz_validation_shard_00369.parquet')
    assert 369 not in manifest['training_indices']
    assert len(set(manifest['training_indices'])) == 520
    assert prepare_data(tmp_path / 'cache', stage, 520, source) == manifest
    pq.write_table(pa.table({'text': ['unexpected']}), stage / 'stray.parquet')
    with pytest.raises(ValueError, match='Unexpected shards'):
        prepare_data(tmp_path / 'cache', stage, 520, source)


@pytest.mark.parametrize('model_type', ['gpt', 'bigram_engram_gpt', 'stemgpt_350m', 'qwen3_0p5b', 'qwen3_0p5b_stem'])
def test_new_checkpoint_round_trip(tmp_path, monkeypatch, model_type):
    from nanochat import checkpoint_manager as checkpoints
    from nanochat.gpt import GPTConfig
    config = GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=1, n_embd=32, head_dim=16, window_pattern='L', stem_layers=[1] if model_type == 'qwen3_0p5b_stem' else [])
    model = checkpoints._resolve_model_class(model_type)(config)
    model.init_weights()
    state = model.state_dict()
    checkpoints.save_checkpoint(str(tmp_path), 1, state, None, {'model_type': model_type, 'model_config': asdict(config)})
    class Tokenizer:
        def get_vocab_size(self):
            return 128
    monkeypatch.setattr(checkpoints, 'get_tokenizer', Tokenizer)
    restored, _, _ = checkpoints.build_model(str(tmp_path), 1, torch.device('cpu'), 'eval')
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value.float(), state[key].float(), rtol=0, atol=0)


@pytest.mark.parametrize('factor,experts', [(6, 6), (12, 6), (6, 12), (12, 12), (24, 6), (6, 24)])
def test_shared_bigram_memory_backward(factor, experts):
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(sequence_len=16, vocab_size=128, n_layer=4, n_head=4, n_kv_head=4, n_embd=64, window_pattern='L', value_embeds_layers=[1, 3], moe_ve_enabled=True, moe_ve_setting=f'0_2_{experts}', moe_ve_gate_nl='sigmoid-norm', moe_ve_router_input='hidden', moe_ve_bias_update='none', moe_ve_balance_lr=0, moe_ve_network_shared_table=True, moe_ve_slot_index_mode='bigram', moe_ve_bigram_slot_factor=factor, moe_ve_slot_vocab_size=128 * factor)
    model = GPT(config)
    model.init_weights()
    memory = model.embed_value_moe
    assert memory.layers[0].routed_embedding is memory.layers[1].routed_embedding
    idx = torch.randint(0, 128, (2, 8))
    loss = model(idx, torch.randint(0, 128, (2, 8)))
    loss.backward()
    assert torch.isfinite(loss)
    grad = memory.network_shared_routed_embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


@pytest.mark.parametrize('model_type', ['gpt', 'stemgpt_350m', 'qwen3_0p5b'])
@pytest.mark.parametrize('memory_type', ['dense', 'mome'])
def test_value_table_weight_decay_reaches_optimizer(model_type, memory_type):
    from nanochat.checkpoint_manager import _resolve_model_class
    from nanochat.gpt import GPTConfig
    config = GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=1, n_embd=32, head_dim=16, window_pattern='L', stem_layers=[], value_embeds_layers=[1], dense_ve_enabled=memory_type == 'dense', moe_ve_enabled=memory_type == 'mome', moe_ve_setting='0_2_6', moe_ve_gate_nl='sigmoid-norm', moe_ve_router_input='hidden', moe_ve_bias_update='none', moe_ve_balance_lr=0)
    model = _resolve_model_class(model_type)(config)
    model.init_weights()
    optimizer, _ = model.setup_optimizers(value_embedding_lr=0.2, value_embedding_adam_betas=(0.8, 0.995), value_embedding_adam_eps=1e-10, value_embedding_weight_decay=0.001)
    tables = {id(p) for p in model.parameters() if getattr(p, 'value_embedding_is_table', False)}
    checked = set()
    for group in optimizer.param_groups:
        matches = tables & {id(p) for p in group['params']}
        if matches:
            assert group['weight_decay'] == 0.001
            assert group['betas'] == (0.8, 0.995)
            assert group['eps'] == 1e-10
            checked |= matches
    assert tables and checked == tables
