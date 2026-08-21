"""Build a bounded, disk-backed Cangjie token-shard training set."""

from concurrent.futures import ProcessPoolExecutor

from .config import (
    block_size,
    dataset_mix,
    dataset_name,
    dataset_split,
    preprocess_queue_depth,
    preprocess_workers,
    streaming_shuffle_buffer,
    streaming_text_batch_size,
    token_shard_dir,
    token_shard_max_cache_gb,
    token_shard_size_mb,
)
from .dataset import (
    StreamingCangjieDataset,
    TokenShardWriter,
    _iter_stream_text_batches,
    _process_text_batch,
)


def _processed_text_batches(stream, text_batch_size):
    """Keep a small ordered process queue so network reading and tokenization overlap."""
    text_batches = iter(_iter_stream_text_batches(stream, text_batch_size))
    with ProcessPoolExecutor(max_workers=preprocess_workers) as executor:
        pending = {}
        next_submit = 0
        next_write = 0
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < preprocess_queue_depth:
                try:
                    texts = next(text_batches)
                except StopIteration:
                    exhausted = True
                    break
                pending[next_submit] = executor.submit(_process_text_batch, texts)
                next_submit += 1

            if next_write not in pending:
                continue
            yield pending.pop(next_write).result()
            next_write += 1


def main():
    source_metadata = {
        "dataset_name": dataset_name,
        "dataset_mix": dataset_mix,
        "dataset_split": dataset_split,
        "shuffle_buffer": streaming_shuffle_buffer,
        "text_batch_size": streaming_text_batch_size,
    }
    writer = TokenShardWriter(
        token_shard_dir,
        shard_size_mb=token_shard_size_mb,
        max_cache_gb=token_shard_max_cache_gb,
        block_size=block_size,
        metadata=source_metadata,
    )
    if writer.remaining_rows == 0:
        print(f"token shard cache 已達上限：{writer.rows_written:,} rows")
        return

    stream_source = StreamingCangjieDataset(
        dataset_name=dataset_name,
        dataset_mix=dataset_mix,
        split=dataset_split,
        block_size=block_size,
        shuffle_buffer=streaming_shuffle_buffer,
        text_batch_size=streaming_text_batch_size,
    )
    stream = stream_source._build_stream()
    rows_to_skip = writer.rows_written
    if rows_to_skip:
        print(f"續跑預處理：會重新讀取並略過前 {rows_to_skip:,} token rows")
    print(
        "開始建立 token shards："
        f"dir={token_shard_dir} | max={token_shard_max_cache_gb}GB | "
        f"shard={token_shard_size_mb}MB | workers={preprocess_workers}"
    )

    processed_batches = 0
    for rows, target_ids in _processed_text_batches(stream, streaming_text_batch_size):
        if rows_to_skip:
            skip = min(rows_to_skip, len(rows))
            rows = rows[skip:]
            target_ids = target_ids[skip:]
            rows_to_skip -= skip
            if len(rows) == 0:
                continue

        writer.append(rows, target_ids)
        processed_batches += 1
        if processed_batches % 32 == 0:
            cache_gb = writer.total_rows * writer.bytes_per_row / 1024**3
            print(f"  已處理 {writer.total_rows:,} rows ({cache_gb:.2f} GB)")
        if writer.remaining_rows == 0:
            break

    writer.finish()
    cache_gb = writer.rows_written * writer.bytes_per_row / 1024**3
    print(
        f"token shards 完成：{writer.rows_written:,} rows | "
        f"{len(writer.manifest['shards'])} shards | {cache_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()
