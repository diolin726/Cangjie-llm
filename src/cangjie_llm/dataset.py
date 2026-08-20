import json
import math
import os
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import torch
from torch.utils.data import Dataset

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
            from datasets import interleave_datasets, load_dataset

            if json_path is not None:
                ds = load_dataset("json", data_files=json_path, split=split)
            elif dataset_mix:
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
                    )
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
                    stopping_strategy="all_exhausted",
                )
                print(
                    "資料集加權混合: "
                    + ", ".join(
                        f"{item['name']}={prob:.1%}"
                        for item, prob in zip(dataset_mix, probabilities)
                    )
                )
            else:
                ds = load_dataset(
                    dataset_name,
                    data_dir=dataset_dir,
                    data_files=data_files,
                    split=split,
                )
            if "source" in ds.column_names:
                all_sources = sorted(
                    str(source) for source in ds.unique("source") if source is not None
                )
                print(f"所有 source: {all_sources}")
            if source_filter:
                before_filter = len(ds)
                if source_filter_mode == "include":
                    ds = ds.filter(lambda row: row.get("source") == source_filter)
                else:
                    ds = ds.filter(lambda row: row.get("source") != source_filter)
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
            print(ds[0])
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


def collate_cangjie_batch(batch):
    xs, targets = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(targets, dim=0)


__all__ = ["CangjieDataset", "collate_cangjie_batch"]
