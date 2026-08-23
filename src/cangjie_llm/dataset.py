import json
import math
import os
import shutil
import tempfile
from bisect import bisect_right
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from cangjie_convertor._shared import get_cj_key_fingerprint

from .config import (
    block_size as default_block_size,
    cache_format_version,
    preprocess_batch_size,
    preprocess_chunk_rows,
    preprocess_queue_depth,
    preprocess_workers,
    training_escape_pattern,
    window_stride,
)
from .tokenization import tokenizer

_preprocess_state = {}


def _normalize_cache_value(value):
    if isinstance(value, dict):
        return {key: _normalize_cache_value(val) for key, val in sorted(value.items())}
    if isinstance(value, tuple):
        return [_normalize_cache_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_cache_value(item) for item in value]
    return value


def _build_cache_metadata(
    dataset_name,
    dataset_mix,
    dataset_dir,
    data_files,
    json_path,
    split,
    cj_key_fingerprint,
    source_filter,
    source_filter_mode,
):
    return {
        "cache_format_version": cache_format_version,
        "dataset_name": dataset_name,
        "dataset_mix": _normalize_cache_value(dataset_mix),
        "dataset_dir": dataset_dir,
        "data_files": _normalize_cache_value(data_files),
        "json_path": json_path,
        "split": split,
        "cj_key_fingerprint": cj_key_fingerprint,
        "source_filter": source_filter,
        "source_filter_mode": source_filter_mode,
    }


def _pack_rows_np(rows):
    rows64 = rows.astype(np.int64, copy=False)
    return (
        (rows64[:, 0] << 36)
        | (rows64[:, 1] << 27)
        | (rows64[:, 2] << 18)
        | (rows64[:, 3] << 9)
        | rows64[:, 4]
    )


def _build_output_lookup_arrays():
    tok = tokenizer()
    codes = tok.all_vocab().cpu().numpy().astype(np.int64, copy=False)
    hashes = _pack_rows_np(codes)
    order = np.argsort(hashes, kind="mergesort")
    return hashes[order], order.astype(np.int16, copy=False)


def _rows_to_output_ids(rows, sorted_hashes, sorted_indices):
    row_hashes = _pack_rows_np(rows)
    positions = np.searchsorted(sorted_hashes, row_hashes)
    return sorted_indices[positions]


def _init_preprocess_worker():
    import opencc

    tok = tokenizer()
    pad_id = tok.vocab["[PAD]"]
    bos_row = np.array(
        [tok.vocab["[BOS]"], pad_id, pad_id, pad_id, pad_id],
        dtype=np.int16,
    )
    eos_row = np.array(
        [tok.vocab["[EOS]"], pad_id, pad_id, pad_id, pad_id],
        dtype=np.int16,
    )
    sorted_hashes, sorted_indices = _build_output_lookup_arrays()
    _preprocess_state["converter"] = opencc.OpenCC("s2twp")
    _preprocess_state["tokenizer"] = tok
    _preprocess_state["bos_row"] = bos_row
    _preprocess_state["eos_row"] = eos_row
    _preprocess_state["sorted_hashes"] = sorted_hashes
    _preprocess_state["sorted_indices"] = sorted_indices


def _process_text_batch(texts):
    if not _preprocess_state:
        _init_preprocess_worker()

    converter = _preprocess_state["converter"]
    tok = _preprocess_state["tokenizer"]
    bos_row = _preprocess_state["bos_row"]
    eos_row = _preprocess_state["eos_row"]
    sorted_hashes = _preprocess_state["sorted_hashes"]
    sorted_indices = _preprocess_state["sorted_indices"]

    row_parts = []
    total_rows = 0
    for text in texts:
        cleaned = training_escape_pattern.sub(" ", text or "")
        normalized = converter.convert(cleaned)
        token_rows = tok.tokenize(normalized)
        row_count = len(token_rows) + 2
        doc_rows = np.empty((row_count, 5), dtype=np.int16)
        doc_rows[0] = bos_row
        if token_rows:
            doc_rows[1:-1] = np.asarray(token_rows, dtype=np.int16)
        doc_rows[-1] = eos_row
        row_parts.append(doc_rows)
        total_rows += row_count

    if not row_parts:
        empty_rows = np.empty((0, 5), dtype=np.int16)
        empty_ids = np.empty((0,), dtype=np.int16)
        return empty_rows, empty_ids

    rows = np.concatenate(row_parts, axis=0) if len(row_parts) > 1 else row_parts[0]
    output_ids = _rows_to_output_ids(rows, sorted_hashes, sorted_indices)
    return rows, output_ids


def _iter_text_batches(ds, total_rows):
    for batch_idx, start in enumerate(range(0, total_rows, preprocess_batch_size)):
        batch = ds[start:start + preprocess_batch_size]
        if isinstance(batch, dict):
            texts = batch.get("text", [])
        else:
            texts = batch
        yield batch_idx, texts


def _load_source_dataset(
    json_path=None,
    dataset_name=None,
    dataset_mix=None,
    dataset_dir=None,
    data_files=None,
    split="train",
    streaming=False,
    verbose=True,
):
    from datasets import interleave_datasets, load_dataset

    if json_path is not None:
        return load_dataset("json", data_files=json_path, split=split, streaming=streaming)

    if dataset_mix:
        loaded_datasets = []
        weights = []
        for mix_item in dataset_mix:
            mix_name = mix_item["name"]
            mix_split = mix_item.get("split", split)
            mix_dataset_dir = mix_item.get("dataset_dir")
            mix_data_files = mix_item.get("data_files")
            mix_weight = float(mix_item.get("weight", 1.0))
            loaded = load_dataset(
                mix_name,
                data_dir=mix_dataset_dir,
                data_files=mix_data_files,
                split=mix_split,
                streaming=streaming,
            )
            if streaming and verbose:
                print(
                    f"載入混合資料集(Streaming): {mix_name} | split={mix_split} | "
                    f"weight={mix_weight:.3f}"
                )
            else:
                print(
                    f"載入混合資料集: {mix_name} | split={mix_split} | "
                    f"weight={mix_weight:.3f} | rows={len(loaded):,}"
                )
            loaded_datasets.append(loaded)
            weights.append(mix_weight)

        weight_sum = sum(weights)
        if weight_sum <= 0:
            raise ValueError("dataset_mix 的 weight 總和必須大於 0")
        probabilities = [weight / weight_sum for weight in weights]
        ds = interleave_datasets(
            loaded_datasets,
            probabilities=probabilities,
            seed=67,
            stopping_strategy="first_exhausted",
        )
        if verbose:
            print(
                "資料集加權混合: "
                + ", ".join(
                    f"{item['name']}={prob:.1%}"
                    for item, prob in zip(dataset_mix, probabilities)
                )
            )
        return ds

    return load_dataset(
        dataset_name,
        data_dir=dataset_dir,
        data_files=data_files,
        split=split,
        streaming=streaming,
    )


def _apply_source_filter(ds, source_filter=None, source_filter_mode="include", streaming=False):
    if not source_filter:
        return ds

    if source_filter_mode == "include":
        predicate = lambda row: row.get("source") == source_filter
    else:
        predicate = lambda row: row.get("source") != source_filter

    if streaming:
        print(f"資料來源過濾(Streaming): {source_filter_mode}={source_filter}")
        return ds.filter(predicate)

    before_filter = len(ds)
    ds = ds.filter(predicate)
    after_filter = len(ds)
    print(
        f"資料來源過濾: {source_filter_mode}={source_filter} | "
        f"{before_filter:,} -> {after_filter:,} 筆"
    )
    if after_filter == 0:
        raise ValueError(
            f"找不到 source={source_filter!r} 的資料，"
            "請確認資料集的 source 欄位名稱。"
        )
    return ds


def _preview_first_row(ds, streaming=False):
    if streaming:
        sample_iter = iter(ds.take(1))
        try:
            print(next(sample_iter))
        except StopIteration:
            print("Streaming 資料集為空")
        return
    print(ds[0])


def _iter_stream_text_batches(ds, batch_size):
    texts = []
    for row in ds:
        texts.append((row or {}).get("text", ""))
        if len(texts) >= batch_size:
            yield texts
            texts = []
    if texts:
        yield texts


class CangjieDataset(Dataset):  # this part is by ai, im sorry but im trash
    def __init__(
        self,
        json_path=None,
        dataset_name=None,
        dataset_mix=None,
        dataset_dir=None,
        data_files=None,
        split="train",
        block_size=default_block_size,
        cache_path=None,
        source_filter=None,
        source_filter_mode="include",
    ):
        self.block_size = block_size
        if source_filter_mode not in {"include", "exclude"}:
            raise ValueError("source_filter_mode 必須是 'include' 或 'exclude'")
        self.cj_key_fingerprint = get_cj_key_fingerprint()
        self.cache_metadata = _build_cache_metadata(
            dataset_name=dataset_name,
            dataset_mix=dataset_mix,
            dataset_dir=dataset_dir,
            data_files=data_files,
            json_path=json_path,
            split=split,
            cj_key_fingerprint=self.cj_key_fingerprint,
            source_filter=source_filter,
            source_filter_mode=source_filter_mode,
        )
        self.cache_meta_path = self._get_cache_meta_path(cache_path)
        self.target_cache_path = self._get_target_cache_path(cache_path)
        data_cache_valid = cache_path and os.path.exists(cache_path) and self._cache_matches_expected()
        if cache_path and os.path.exists(cache_path) and not data_cache_valid:
            print("現有 data 快取與目前資料設定不符，將重新建立")

        if data_cache_valid:
            print(f"從快取載入: {cache_path}")
            self.data = torch.load(cache_path, weights_only=True)
            print(
                "載入完成: "
                f"shape={self.data.shape}, "
                f"記憶體={self.data.element_size() * self.data.nelement() / 1024**3:.2f} GB"
            )
        else:
            print("首次預處理（後續會從快取載入）...")
            ds = _load_source_dataset(
                json_path=json_path,
                dataset_name=dataset_name,
                dataset_mix=dataset_mix,
                dataset_dir=dataset_dir,
                data_files=data_files,
                split=split,
                streaming=False,
            )
            if "source" in ds.column_names:
                all_sources = sorted(
                    str(source) for source in ds.unique("source") if source is not None
                )
                print(f"所有 source: {all_sources}")
            ds = _apply_source_filter(
                ds,
                source_filter=source_filter,
                source_filter_mode=source_filter_mode,
                streaming=False,
            )
            _preview_first_row(ds, streaming=False)
            total_rows = len(ds)
            self.data, self.target_ids = self._preprocess_dataset(ds, total_rows)
            print(
                "預處理完成: "
                f"shape={self.data.shape}, "
                f"記憶體={self.data.element_size() * self.data.nelement() / 1024**3:.2f} GB"
            )
            print(
                f"target ids 完成: shape={self.target_ids.shape}, "
                f"記憶體={self.target_ids.element_size() * self.target_ids.nelement() / 1024**3:.2f} GB"
            )
            if cache_path:
                torch.save(self.data, cache_path)
                print(f"已儲存快取: {cache_path}")
            if self.target_cache_path:
                torch.save(self.target_ids, self.target_cache_path)
                print(f"已儲存 target id 快取: {self.target_cache_path}")
            self._write_cache_meta()

        if (
            not hasattr(self, "target_ids")
            and self.target_cache_path
            and os.path.exists(self.target_cache_path)
            and self._cache_matches_expected()
        ):
            print(f"從快取載入 target ids: {self.target_cache_path}")
            self.target_ids = torch.load(self.target_cache_path, weights_only=True)
            print(
                f"target ids 載入完成: shape={self.target_ids.shape}, "
                f"記憶體={self.target_ids.element_size() * self.target_ids.nelement() / 1024**3:.2f} GB"
            )
        elif not hasattr(self, "target_ids"):
            self.target_ids = None

        expected_target_len = len(self.data) - 1
        if self.target_ids is not None and len(self.target_ids) != expected_target_len:
            print(
                f"target ids 快取長度不符，預期 {expected_target_len}，"
                f"實際 {len(self.target_ids)}，將重新建立"
            )
            self.target_ids = None

        if self.target_ids is None:
            print("建立 target id 快取...")
            self.target_ids = self._build_target_ids()
            print(
                f"target ids 完成: shape={self.target_ids.shape}, "
                f"記憶體={self.target_ids.element_size() * self.target_ids.nelement() / 1024**3:.2f} GB"
            )
            if self.target_cache_path:
                torch.save(self.target_ids, self.target_cache_path)
                self._write_cache_meta()
                print(f"已儲存 target id 快取: {self.target_cache_path}")

    @staticmethod
    def _get_cache_meta_path(cache_path):
        if cache_path is None:
            return None
        root, _ = os.path.splitext(cache_path)
        return f"{root}.meta.json"

    @staticmethod
    def _get_target_cache_path(cache_path):
        if cache_path is None:
            return None
        root, ext = os.path.splitext(cache_path)
        return f"{root}_target_ids{ext or '.pt'}"

    def _cache_matches_expected(self):
        if self.cache_meta_path is None or not os.path.exists(self.cache_meta_path):
            if self.cache_metadata.get("split") not in (None, "train"):
                return False
            return True
        with open(self.cache_meta_path, "r", encoding="utf-8") as file:
            meta = json.load(file)
        return meta == self.cache_metadata

    def _write_cache_meta(self):
        if self.cache_meta_path is None:
            return
        with open(self.cache_meta_path, "w", encoding="utf-8") as file:
            json.dump(self.cache_metadata, file, ensure_ascii=False, indent=2)

    def _flush_preprocess_chunk(self, temp_dir, chunk_index, row_parts, output_id_parts):
        rows = np.concatenate(row_parts, axis=0) if len(row_parts) > 1 else row_parts[0]
        output_ids = (
            np.concatenate(output_id_parts, axis=0)
            if len(output_id_parts) > 1
            else output_id_parts[0]
        )
        data_chunk_path = os.path.join(temp_dir, f"data_chunk_{chunk_index:05d}.npy")
        output_chunk_path = os.path.join(temp_dir, f"output_chunk_{chunk_index:05d}.npy")
        np.save(data_chunk_path, rows)
        np.save(output_chunk_path, output_ids)
        return data_chunk_path, output_chunk_path, len(rows)

    def _preprocess_dataset(self, ds, total_rows):
        temp_dir = tempfile.mkdtemp(
            prefix="cangjie-preprocess-",
            dir=os.path.dirname(os.path.abspath(self.cache_meta_path or ".")),
        )
        chunk_records = []
        pending_results = {}
        row_parts = []
        output_id_parts = []
        chunk_row_count = 0
        total_token_rows = 0
        chunk_index = 0
        next_batch_to_write = 0

        def flush_pending_chunk():
            nonlocal row_parts, output_id_parts, chunk_row_count, chunk_index
            if not row_parts:
                return
            chunk_records.append(
                self._flush_preprocess_chunk(temp_dir, chunk_index, row_parts, output_id_parts)
            )
            chunk_index += 1
            row_parts = []
            output_id_parts = []
            chunk_row_count = 0

        try:
            batch_iter = iter(_iter_text_batches(ds, total_rows))
            max_workers = min(preprocess_workers, max(1, math.ceil(total_rows / preprocess_batch_size)))
            if max_workers <= 1:
                for batch_idx, texts in batch_iter:
                    pending_results[batch_idx] = _process_text_batch(texts)
                    while next_batch_to_write in pending_results:
                        batch_rows, batch_output_ids = pending_results.pop(next_batch_to_write)
                        row_parts.append(batch_rows)
                        output_id_parts.append(batch_output_ids)
                        chunk_row_count += len(batch_rows)
                        total_token_rows += len(batch_rows)
                        processed = min((next_batch_to_write + 1) * preprocess_batch_size, total_rows)
                        if processed % 10000 == 0 or processed == total_rows:
                            print(f"  已處理 {processed}/{total_rows} 篇，共 {total_token_rows} tokens")
                        if chunk_row_count >= preprocess_chunk_rows:
                            flush_pending_chunk()
                        next_batch_to_write += 1
            else:
                with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_preprocess_worker) as executor:
                    in_flight = {}
                    batch_iter_exhausted = False
                    while not batch_iter_exhausted or in_flight:
                        while not batch_iter_exhausted and len(in_flight) < preprocess_queue_depth:
                            try:
                                batch_idx, texts = next(batch_iter)
                            except StopIteration:
                                batch_iter_exhausted = True
                                break
                            in_flight[executor.submit(_process_text_batch, texts)] = batch_idx

                        done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                        for future in done:
                            batch_idx = in_flight.pop(future)
                            pending_results[batch_idx] = future.result()

                        while next_batch_to_write in pending_results:
                            batch_rows, batch_output_ids = pending_results.pop(next_batch_to_write)
                            row_parts.append(batch_rows)
                            output_id_parts.append(batch_output_ids)
                            chunk_row_count += len(batch_rows)
                            total_token_rows += len(batch_rows)
                            processed = min((next_batch_to_write + 1) * preprocess_batch_size, total_rows)
                            if processed % 10000 == 0 or processed == total_rows:
                                print(f"  已處理 {processed}/{total_rows} 篇，共 {total_token_rows} tokens")
                            if chunk_row_count >= preprocess_chunk_rows:
                                flush_pending_chunk()
                            next_batch_to_write += 1

            flush_pending_chunk()

            total_output_ids = max(0, total_token_rows - 1)
            data_tensor = torch.empty((total_token_rows, 5), dtype=torch.int16)
            target_tensor = torch.empty(total_output_ids, dtype=torch.int16)

            data_offset = 0
            target_offset = 0
            skip_first_output = True
            for data_chunk_path, output_chunk_path, row_count in chunk_records:
                rows = np.load(data_chunk_path)
                output_ids = np.load(output_chunk_path)
                data_tensor[data_offset:data_offset + row_count] = torch.from_numpy(rows)
                data_offset += row_count

                start_idx = 1 if skip_first_output else 0
                if len(output_ids) > start_idx:
                    chunk_target = output_ids[start_idx:]
                    target_len = len(chunk_target)
                    target_tensor[target_offset:target_offset + target_len] = torch.from_numpy(chunk_target)
                    target_offset += target_len
                skip_first_output = False

            return data_tensor, target_tensor
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _build_target_ids(self):
        tok = tokenizer()
        codes = tok.all_vocab().to(torch.long)
        hashes = (
            (codes[:, 0] << 36)
            | (codes[:, 1] << 27)
            | (codes[:, 2] << 18)
            | (codes[:, 3] << 9)
            | codes[:, 4]
        )
        sorted_hashes, sorted_indices = torch.sort(hashes)

        target_long = self.data[1:].to(torch.long)
        target_hash = (
            (target_long[:, 0] << 36)
            | (target_long[:, 1] << 27)
            | (target_long[:, 2] << 18)
            | (target_long[:, 3] << 9)
            | target_long[:, 4]
        )
        pos = torch.searchsorted(sorted_hashes, target_hash)
        return sorted_indices[pos].to(torch.int16)

    def __len__(self):
        usable_len = min(len(self.data), len(self.target_ids))
        return max(0, (usable_len - self.block_size) // window_stride + 1)

    def __getitem__(self, idx):
        idx = idx * window_stride
        # Return owned tensors so DataLoader workers can collate safely.
        x = self.data[idx:idx + self.block_size].clone()
        target = self.target_ids[idx:idx + self.block_size].clone()
        return x, target


class StreamingCangjieDataset(IterableDataset):
    is_streaming = True

    def __init__(
        self,
        json_path=None,
        dataset_name=None,
        dataset_mix=None,
        dataset_dir=None,
        data_files=None,
        split="train",
        block_size=default_block_size,
        source_filter=None,
        source_filter_mode="include",
        shuffle_buffer=10_000,
        text_batch_size=128,
        seed=67,
    ):
        self.json_path = json_path
        self.dataset_name = dataset_name
        self.dataset_mix = dataset_mix
        self.dataset_dir = dataset_dir
        self.data_files = data_files
        self.split = split
        self.block_size = block_size
        self.source_filter = source_filter
        self.source_filter_mode = source_filter_mode
        self.shuffle_buffer = shuffle_buffer
        self.text_batch_size = text_batch_size
        self.seed = seed
        self.epoch = 0

        if source_filter_mode not in {"include", "exclude"}:
            raise ValueError("source_filter_mode 必須是 'include' 或 'exclude'")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _build_stream(self, worker_id=0, num_workers=1):
        ds = _load_source_dataset(
            json_path=self.json_path,
            dataset_name=self.dataset_name,
            dataset_mix=self.dataset_mix,
            dataset_dir=self.dataset_dir,
            data_files=self.data_files,
            split=self.split,
            streaming=True,
            verbose=worker_id == 0,
        )
        ds = _apply_source_filter(
            ds,
            source_filter=self.source_filter,
            source_filter_mode=self.source_filter_mode,
            streaming=True,
        )
        if num_workers > 1:
            # Give each worker a separate source slice before it shuffles and tokenizes.
            ds = ds.shard(num_shards=num_workers, index=worker_id)
        if self.shuffle_buffer and self.shuffle_buffer > 0:
            ds = ds.shuffle(
                seed=self.seed + self.epoch + worker_id,
                buffer_size=self.shuffle_buffer,
            )
        return ds

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        if worker_id != 0:
            from datasets import disable_progress_bars

            disable_progress_bars()
        ds = self._build_stream(worker_id=worker_id, num_workers=num_workers)

        row_buffer = np.empty((0, 5), dtype=np.int16)
        output_buffer = np.empty((0,), dtype=np.int16)
        consumed = 0

        for texts in _iter_stream_text_batches(ds, self.text_batch_size):
            batch_rows, batch_output_ids = _process_text_batch(texts)
            if len(batch_rows) == 0:
                continue

            if len(row_buffer) == 0:
                row_buffer = batch_rows
                output_buffer = batch_output_ids
            else:
                row_buffer = np.concatenate((row_buffer, batch_rows), axis=0)
                output_buffer = np.concatenate((output_buffer, batch_output_ids), axis=0)

            while consumed + self.block_size + 1 <= len(row_buffer):
                x = torch.from_numpy(
                    row_buffer[consumed:consumed + self.block_size].copy()
                )
                target = torch.from_numpy(
                    output_buffer[consumed + 1:consumed + 1 + self.block_size].copy()
                )
                yield x, target
                consumed += window_stride

            if consumed > 0:
                row_buffer = row_buffer[consumed:]
                output_buffer = output_buffer[consumed:]
                consumed = 0


class TokenShardWriter:
    """Write aligned Cangjie input/output rows into resumable fixed-size shards."""

    manifest_name = "manifest.json"
    format_version = 1
    bytes_per_row = 12  # Five int16 Cangjie codes plus one int16 output id.

    def __init__(self, shard_dir, shard_size_mb, max_cache_gb, block_size, metadata=None):
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.shard_dir / self.manifest_name
        self.shard_size_bytes = int(shard_size_mb * 1024 * 1024)
        self.max_rows = int(max_cache_gb * 1024 * 1024 * 1024) // self.bytes_per_row
        self.rows_per_shard = max(1, self.shard_size_bytes // self.bytes_per_row)
        self.block_size = block_size
        self.manifest = self._load_or_create_manifest(metadata or {})
        self.rows_written = int(self.manifest["rows_written"])
        self._rows = None
        self._target_ids = None
        self._buffered_rows = 0

    def _load_or_create_manifest(self, metadata):
        if not self.manifest_path.exists():
            return {
                "format_version": self.format_version,
                "block_size": self.block_size,
                "rows_per_shard": self.rows_per_shard,
                "max_rows": self.max_rows,
                "rows_written": 0,
                "shards": [],
                "source": metadata,
            }

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format_version") != self.format_version:
            raise ValueError("token shard manifest 格式不相容，請使用新的目錄")
        if manifest.get("block_size") != self.block_size:
            raise ValueError("token shard 的 block_size 與目前設定不一致")
        if manifest.get("rows_per_shard") != self.rows_per_shard:
            raise ValueError("token shard 大小與目前設定不一致")
        for shard in manifest.get("shards", []):
            if not (self.shard_dir / shard["tokens"]).exists() or not (
                self.shard_dir / shard["targets"]
            ).exists():
                raise FileNotFoundError(f"缺少 token shard: {shard}")
        return manifest

    @property
    def remaining_rows(self):
        return max(0, self.max_rows - self.rows_written - self._buffered_rows)

    @property
    def total_rows(self):
        return self.rows_written + self._buffered_rows

    def append(self, rows, target_ids):
        if len(rows) != len(target_ids):
            raise ValueError("rows 與 target_ids 長度必須一致")
        offset = 0
        while offset < len(rows) and self.remaining_rows > 0:
            if self._rows is None:
                self._rows = np.empty((self.rows_per_shard, 5), dtype=np.int16)
                self._target_ids = np.empty((self.rows_per_shard,), dtype=np.int16)
                self._buffered_rows = 0

            take = min(
                len(rows) - offset,
                self.rows_per_shard - self._buffered_rows,
                self.remaining_rows,
            )
            end = self._buffered_rows + take
            self._rows[self._buffered_rows:end] = rows[offset:offset + take]
            self._target_ids[self._buffered_rows:end] = target_ids[offset:offset + take]
            self._buffered_rows = end
            offset += take
            if self._buffered_rows == self.rows_per_shard:
                self._flush_shard()
        return offset

    def finish(self):
        if self._buffered_rows:
            self._flush_shard()

    def _flush_shard(self):
        shard_index = len(self.manifest["shards"])
        token_name = f"shard_{shard_index:05d}_tokens.npy"
        target_name = f"shard_{shard_index:05d}_targets.npy"
        token_path = self.shard_dir / token_name
        target_path = self.shard_dir / target_name
        token_tmp_path = self.shard_dir / f".{token_name}.tmp"
        target_tmp_path = self.shard_dir / f".{target_name}.tmp"
        row_count = self._buffered_rows
        with token_tmp_path.open("wb") as handle:
            np.save(handle, self._rows[:row_count])
        with target_tmp_path.open("wb") as handle:
            np.save(handle, self._target_ids[:row_count])
        os.replace(token_tmp_path, token_path)
        os.replace(target_tmp_path, target_path)
        self.manifest["shards"].append(
            {"tokens": token_name, "targets": target_name, "rows": row_count}
        )
        self.rows_written += row_count
        self.manifest["rows_written"] = self.rows_written
        self._write_manifest()
        self._rows = None
        self._target_ids = None
        self._buffered_rows = 0

    def _write_manifest(self):
        tmp_path = self.manifest_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.manifest_path)


class TokenShardDataset(Dataset):
    """Random-access training dataset backed by memory-mapped token shards."""

    def __init__(
        self,
        shard_dir,
        block_size=default_block_size,
        stride=window_stride,
        shard_start=0,
        shard_stop=None,
    ):
        self.shard_dir = Path(shard_dir)
        manifest_path = self.shard_dir / TokenShardWriter.manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(f"找不到 token shard manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format_version") != TokenShardWriter.format_version:
            raise ValueError("token shard manifest 格式不相容")
        if manifest.get("block_size") != block_size:
            raise ValueError("token shard 的 block_size 與目前設定不一致")

        self.block_size = block_size
        self.stride = stride
        all_shards = manifest.get("shards", [])
        self.total_shard_count = len(all_shards)
        self.shards = all_shards[shard_start:shard_stop]
        if not self.shards:
            raise ValueError("選取的 token shards 為空")
        self._counts = [
            max(0, (int(shard["rows"]) - block_size - 1) // stride + 1)
            for shard in self.shards
        ]
        self._cumulative_counts = np.cumsum(self._counts).tolist()
        self._opened = {}

    def __len__(self):
        return self._cumulative_counts[-1] if self._cumulative_counts else 0

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._cumulative_counts, index)
        previous_count = self._cumulative_counts[shard_index - 1] if shard_index else 0
        row_start = (index - previous_count) * self.stride
        rows, target_ids = self._open_shard(shard_index)
        x = torch.from_numpy(np.array(rows[row_start:row_start + self.block_size], copy=True))
        target = torch.from_numpy(
            np.array(target_ids[row_start + 1:row_start + self.block_size + 1], copy=True)
        )
        return x, target

    def _open_shard(self, shard_index):
        opened = self._opened.get(shard_index)
        if opened is None:
            shard = self.shards[shard_index]
            rows = np.load(self.shard_dir / shard["tokens"], mmap_mode="r")
            target_ids = np.load(self.shard_dir / shard["targets"], mmap_mode="r")
            opened = (rows, target_ids)
            self._opened[shard_index] = opened
        return opened

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_opened"] = {}
        return state


def collate_cangjie_batch(batch):
    xs, targets = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(targets, dim=0)


__all__ = [
    "CangjieDataset",
    "StreamingCangjieDataset",
    "TokenShardDataset",
    "TokenShardWriter",
    "collate_cangjie_batch",
]
