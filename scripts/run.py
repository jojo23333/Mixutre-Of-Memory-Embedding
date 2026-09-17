"""Prepare data and run a setting's seeds sequentially."""

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_assets():
    for line in (ROOT / 'artifacts' / 'SHA256SUMS').read_text().splitlines():
        digest, relative = line.split(None, 1)
        path = ROOT / 'artifacts' / relative.strip()
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Asset checksum mismatch: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='Model family/setting name')
    parser.add_argument('--training-shards', type=int, required=True)
    parser.add_argument('--seeds', default='42', help='Comma-separated seeds')
    parser.add_argument('--nproc-per-node', type=int, default=4)
    parser.add_argument('--attention-backend', choices=('sdpa', 'auto', 'fa3'), default='sdpa')
    parser.add_argument('--device-batch-size', type=int, help='Override microbatch size, preserving total batch tokens')
    parser.add_argument('--plan', action='store_true', help='Print commands without loading PyTorch or downloading data')
    parser.add_argument('--output-dir', type=Path, help='Default: runs/<family>/<setting>')
    parser.add_argument('--data-cache', type=Path, default=ROOT / 'data' / 'fineweb-edu')
    parser.add_argument('--reuse-data', type=Path, help='Reuse downloaded shards from this directory')
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('training_args', nargs=argparse.REMAINDER, help='Training flags after --')
    args = parser.parse_args()
    if len(args.name.split('/')) != 2 or any(part in {'', '.', '..'} for part in args.name.split('/')):
        parser.error('Name must be model-family/setting')
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
        parser.error('Run ID must be a single directory name')
    try:
        seeds = [int(s) for s in args.seeds.split(',')]
    except ValueError:
        parser.error('Seeds must be comma-separated integers')
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error('Seeds must be distinct nonnegative integers')
    if args.nproc_per_node < 1 or (args.device_batch_size is not None and args.device_batch_size < 1):
        parser.error('Process count and device batch size must be positive')
    if not 1 <= args.training_shards <= 1822:
        parser.error('Training shard count must be between 1 and 1822')
    training = args.training_args
    if training[:1] == ['--']:
        training = training[1:]
    if args.device_batch_size is not None:
        training += ['--device-batch-size', str(args.device_batch_size)]

    def value(flag):
        positions = [i for i, arg in enumerate(training) if arg == '--' + flag]
        if not positions or positions[-1] + 1 >= len(training):
            parser.error(f'Missing training option: --{flag}')
        return training[positions[-1] + 1]

    try:
        batch = int(value('device-batch-size'))
        total_batch = int(value('total-batch-size'))
        sequence = int(value('max-seq-len'))
    except ValueError:
        parser.error('Batch sizes and sequence length must be integers')
    if min(batch, total_batch, sequence) < 1 or total_batch % (args.nproc_per_node * batch * sequence):
        parser.error('Total batch must be positive and divisible by processes × device batch × sequence length')
    if '--moe-ve-slot-map-path' in training:
        index = training.index('--moe-ve-slot-map-path') + 1
        training[index] = str(ROOT / value('moe-ve-slot-map-path'))
    commands = []
    for seed in seeds:
        tag = f"{args.name.replace('/', '_')}_{args.run_id}_seed{seed}"
        commands.append([sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={args.nproc_per_node}', '-m', 'scripts.base_train', '--', *training, '--model-tag', tag, '--seed-ids', str(seed)])
    if args.plan:
        for command in commands:
            print(shlex.join(command))
        return

    runtime = (args.output_dir or ROOT / 'runs' / args.name).resolve()
    verify_assets()
    from scripts.prepare_data import prepare_data
    prepare_data(args.data_cache, runtime / 'base_data', args.training_shards, args.reuse_data)
    tokenizer_dir = runtime / 'tokenizer'
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for name in ('tokenizer.pkl', 'token_bytes.pt'):
        source = ROOT / 'artifacts' / 'tokenizer' / name
        target = tokenizer_dir / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f'Refusing to replace a different tokenizer: {target}')
        if not target.exists():
            shutil.copyfile(source, target)
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(runtime), NANOCHAT_FLASH_IMPL=args.attention_backend, PYTHONPATH=str(ROOT))
    env.setdefault('OMP_NUM_THREADS', '1')
    log_dir = runtime / 'logs' / args.run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    for seed, command in zip(seeds, commands):
        log_path = log_dir / f'seed{seed}.log'
        with log_path.open('x') as log:
            with (log_dir / f'seed{seed}.command.txt').open('x') as record:
                record.write(shlex.join(command) + '\n')
            print(shlex.join(command), flush=True)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if status:
            raise SystemExit(f'Seed {seed} failed with exit status {status}; see {log_path}')


if __name__ == '__main__':
    main()
