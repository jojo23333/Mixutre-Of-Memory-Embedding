"""Download and stage a FineWeb-Edu training subset and held-out validation shard."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

BASE_URL = 'https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main'
VALIDATION_SHARD = 369


def training_indices(count):
    if count < 1 or count > 1822:
        raise ValueError('Training shard count must be between 1 and 1822')
    return [i for i in range(count + 1) if i != VALIDATION_SHARD][:count]


def validate_parquet(path):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows == 0 or 'text' not in parquet.schema.names:
        raise ValueError(f'Invalid corpus shard: {path}')
    return parquet.metadata.num_rows


def prepare_data(cache_dir, data_dir, count, reuse_dir=None):
    import requests
    cache_dir, data_dir = Path(cache_dir).resolve(), Path(data_dir).resolve()
    indices = training_indices(count)
    expected = {f'shard_{i:05d}.parquet' for i in indices} | {'zz_validation_shard_00369.parquet'}
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    unexpected = {p.name for p in data_dir.glob('*.parquet')} - expected
    if unexpected:
        raise ValueError(f'Unexpected shards in {data_dir}: {sorted(unexpected)}; use a separate runtime directory')

    def prepare(index):
        name = f'shard_{index:05d}.parquet'
        cache = cache_dir / name
        source = cache
        if not cache.exists() and reuse_dir is not None and (Path(reuse_dir) / name).is_file():
            source = (Path(reuse_dir) / name).resolve()
        if not source.exists():
            temporary = cache.with_suffix('.parquet.part')
            for attempt in range(4):
                try:
                    with requests.get(f'{BASE_URL}/{name}', stream=True, timeout=(20, 120)) as response:
                        response.raise_for_status()
                        with temporary.open('wb') as output:
                            for chunk in response.iter_content(1024 * 1024):
                                output.write(chunk)
                    validate_parquet(temporary)
                    temporary.rename(cache)
                    break
                except (requests.RequestException, OSError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        rows = validate_parquet(source)
        target = data_dir / ('zz_validation_shard_00369.parquet' if index == VALIDATION_SHARD else name)
        if target.is_symlink() or target.exists():
            if target.resolve() != source.resolve():
                raise ValueError(f'Refusing to replace a different shard: {target}')
        else:
            target.symlink_to(source)
        return {'index': index, 'documents': rows, 'bytes': source.stat().st_size}

    with ThreadPoolExecutor(max_workers=4) as pool:
        inventory = list(pool.map(prepare, [*indices, VALIDATION_SHARD]))
    paths = sorted(data_dir.glob('*.parquet'))
    if {p.name for p in paths} != expected or paths[-1].name != 'zz_validation_shard_00369.parquet':
        raise ValueError('Corpus staging does not match the requested training/validation split')
    if paths[-1].resolve() in {p.resolve() for p in paths[:-1]}:
        raise ValueError('The validation shard also appears in training')
    manifest = {'dataset_url': BASE_URL, 'training_indices': indices, 'validation_index': VALIDATION_SHARD, 'inventory': inventory}
    (data_dir / 'corpus_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--training-shards', type=int, required=True)
    parser.add_argument('--reuse-dir', type=Path)
    args = parser.parse_args()
    prepare_data(args.cache_dir, args.data_dir, args.training_shards, args.reuse_dir)


if __name__ == '__main__':
    main()
