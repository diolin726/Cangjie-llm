import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
import numpy as np
import os
import math
import time
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from torch.utils.data import DataLoader, Dataset, Subset
from cangjie_convertor import cj_encoder , cj_decoder
from cangjie_convertor._shared import get_cj_key_fingerprint
from torch.amp import autocast
import threading

torch.manual_seed(67) #676767

# ===== 可調參數 =====
dropout = 0.1
embed_size = 384
vocab_size = 383 #需要手動調整
batch_size = 256
block_size = 256
n_head = 12
n_layer = 12
lr = 3e-5
min_lr = 3e-6
warmup_steps = 200
plateau_patience = 3
plateau_factor = 0.5
plateau_min_delta = 0.003
plateau_min_lr = 3e-6
epochs = 2
log_interval = 1000
checkpoint_interval = 1000
torch_compile_mode = "default"
return_training_logits = False
sampled_softmax_negatives = 0
validation_ratio = 0.002
validation_max_batches = 8
sample_prompts = [""]
sample_max_tokens = 24
preprocess_batch_size = 1024
preprocess_chunk_rows = 10_000_000
preprocess_workers = max(1, min(4, os.cpu_count() or 1))
preprocess_queue_depth = max(2, preprocess_workers * 2)
cache_format_version = 2

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
device = 'cuda' if gpu_count > 0 else 'cpu'
use_bf16_autocast = device == "cuda" and torch.cuda.is_bf16_supported()
enable_torch_compile = (device == "cuda" and gpu_count <= 1)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

print(f"using {device} ({gpu_count} GPU{'s' if gpu_count != 1 else ''} visible)")
if device == "cuda":
    print(f"bf16 autocast: {'enabled' if use_bf16_autocast else 'disabled'}")
    print(f"flash sdp: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"mem efficient sdp: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"math sdp fallback: {torch.backends.cuda.math_sdp_enabled()}")

def get_lr(step, total_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    decay_step = min(max(0, step - warmup_steps), decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
    return min_lr + (lr - min_lr) * cosine


_preprocess_state = {}


def _normalize_cache_value(value):
    if isinstance(value, dict):
        return {key: _normalize_cache_value(val) for key, val in sorted(value.items())}
    if isinstance(value, tuple):
        return [_normalize_cache_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_cache_value(item) for item in value]
    return value


def _build_cache_metadata(dataset_name, dataset_dir, data_files, json_path, split, cj_key_fingerprint):
    return {
        "cache_format_version": cache_format_version,
        "dataset_name": dataset_name,
        "dataset_dir": dataset_dir,
        "data_files": _normalize_cache_value(data_files),
        "json_path": json_path,
        "split": split,
        "cj_key_fingerprint": cj_key_fingerprint,
    }


def _pack_rows_np(rows):
    rows64 = rows.astype(np.int64, copy=False)
    return (
        (rows64[:, 0] << 36) |
        (rows64[:, 1] << 27) |
        (rows64[:, 2] << 18) |
        (rows64[:, 3] << 9)  |
        rows64[:, 4]
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
        normalized = converter.convert(text or "")
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


class CangjieDataset(Dataset): # this part is by ai, im sorry but im trash
    def __init__(self, json_path=None, dataset_name=None , dataset_dir=None , data_files=None , split="train" , block_size=256, cache_path=None):
        self.block_size = block_size
        self.cj_key_fingerprint = get_cj_key_fingerprint()
        self.cache_metadata = _build_cache_metadata(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            data_files=data_files,
            json_path=json_path,
            split=split,
            cj_key_fingerprint=self.cj_key_fingerprint,
        )
        self.cache_meta_path = self._get_cache_meta_path(cache_path)
        self.target_cache_path = self._get_target_cache_path(cache_path)
        data_cache_valid = cache_path and os.path.exists(cache_path) and self._cache_matches_expected()
        if cache_path and os.path.exists(cache_path) and not data_cache_valid:
            print("現有 data 快取與目前資料設定不符，將重新建立")

        if data_cache_valid:
            print(f"從快取載入: {cache_path}")
            self.data = torch.load(cache_path, weights_only=True) #.to(torch.long)
            print(f"載入完成: shape={self.data.shape}, 記憶體={self.data.element_size() * self.data.nelement() / 1024**3:.2f} GB")
        else:
            print("首次預處理（後續會從快取載入）...")
            from datasets import load_dataset

            if json_path is not None :
                ds = load_dataset("json", data_files=json_path, split="train")
            else:
                ds=load_dataset(
                    dataset_name ,
                    data_dir=dataset_dir,
                    data_files=data_files,
                    split=split,
                )
            print(ds[0])
            total_rows = len(ds)
            self.data, self.target_ids = self._preprocess_dataset(ds, total_rows)
            print(f"預處理完成: shape={self.data.shape}, 記憶體={self.data.element_size() * self.data.nelement() / 1024**3:.2f} GB")
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

        if not hasattr(self, "target_ids") and self.target_cache_path and os.path.exists(self.target_cache_path) and self._cache_matches_expected():
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
        #self.data=self.data.to(torch.long)
        #self.data.share_memory_()

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
        with open(self.cache_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta == self.cache_metadata

    def _write_cache_meta(self):
        if self.cache_meta_path is None:
            return
        with open(self.cache_meta_path, "w", encoding="utf-8") as f:
            json.dump(self.cache_metadata, f, ensure_ascii=False, indent=2)

    def _flush_preprocess_chunk(self, temp_dir, chunk_index, row_parts, output_id_parts):
        rows = np.concatenate(row_parts, axis=0) if len(row_parts) > 1 else row_parts[0]
        output_ids = np.concatenate(output_id_parts, axis=0) if len(output_id_parts) > 1 else output_id_parts[0]
        data_chunk_path = os.path.join(temp_dir, f"data_chunk_{chunk_index:05d}.npy")
        output_chunk_path = os.path.join(temp_dir, f"output_chunk_{chunk_index:05d}.npy")
        np.save(data_chunk_path, rows)
        np.save(output_chunk_path, output_ids)
        return data_chunk_path, output_chunk_path, len(rows)

    def _preprocess_dataset(self, ds, total_rows):
        temp_dir = tempfile.mkdtemp(prefix="cangjie-preprocess-", dir=os.path.dirname(os.path.abspath(self.cache_meta_path or ".")))
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
            (codes[:, 0] << 36) |
            (codes[:, 1] << 27) |
            (codes[:, 2] << 18) |
            (codes[:, 3] << 9)  |
            codes[:, 4]
        )
        sorted_hashes, sorted_indices = torch.sort(hashes)

        target_long = self.data[1:].to(torch.long)
        target_hash = (
            (target_long[:, 0] << 36) |
            (target_long[:, 1] << 27) |
            (target_long[:, 2] << 18) |
            (target_long[:, 3] << 9)  |
            target_long[:, 4]
        )
        pos = torch.searchsorted(sorted_hashes, target_hash)
        return sorted_indices[pos].to(torch.int16)

    def __len__(self):
        usable_len = min(len(self.data), len(self.target_ids))
        return max(0, (usable_len - self.block_size) * 2 // self.block_size)

    def __getitem__(self, idx):
        idx = idx * self.block_size // 2
        # Return owned tensors so DataLoader workers can collate safely.
        x = self.data[idx : idx + self.block_size].clone()
        target = self.target_ids[idx : idx + self.block_size].clone()
        return x, target


def collate_cangjie_batch(batch):
    xs, targets = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(targets, dim=0)


class tokenizer():
    def __init__(self):
        self.decoder=cj_decoder()
        self.cj_encoder=cj_encoder(vocab_size)
        self.vocab = self.cj_encoder.vocab
        self.id_to_vocab = self.cj_encoder.id_to_vocab



    def all_vocab(self):
        """建立所有 13228 個輸出詞彙的 5-tuple tensor"""
        PAD = self.vocab["[PAD]"]
        codes = []
        self.output_tokens = []

        # 1) 非 CJK tokens (357個) #383 - 26 = 357
        for token_name, token_id in self.vocab.items():
            if not token_name.startswith('cj_'):
                codes.append([token_id, PAD, PAD, PAD, PAD])
                self.output_tokens.append(token_name)

        # 2) 倉頡排列 (12871個)
        for code in self.decoder.cj_keys:
            ids = [self.vocab[f"cj_{c}"] for c in code]
            ids += [PAD] * (5 - len(ids))
            codes.append(ids)
            self.output_tokens.append(code)

        return torch.tensor(codes, dtype=torch.long)

    def tokenlist_to_id(self , token_list ):
        if(len(token_list) == 5  ):
            return [[self.vocab[t] for t in token_list]]
        if(token_list[0] in self.vocab ):
            return [[ self.vocab[token_list[0]],self.vocab["[PAD]"],self.vocab["[PAD]"],self.vocab["[PAD]"],self.vocab["[PAD]"] ]]

        try:
            utf8_bytes = token_list[0].encode('utf-8')
            return [ [ self.vocab[f"<BYTE_{b}>"] ,self.vocab["[PAD]"],self.vocab["[PAD]"],self.vocab["[PAD]"],self.vocab["[PAD]"]] for b in utf8_bytes]
        except Exception:
            return [[self.vocab["[UNK]"],self.vocab["[PAD]"],self.vocab["[PAD]"],self.vocab["[PAD]"],self.vocab["[PAD]"]]]

    def tokenize(self, s ):
        normalized = unicodedata.normalize('NFKC', s).replace('\u3000', ' ')
        return self.cj_encoder.encode_text_to_rows(normalized)
    def id_decode(self, ids): #id list [26,15,23 ...]
        ans=[]
        for id in ids:
            if id < vocab_size - 26:
                ans.append(self.id_to_vocab[id + 26])
            else:
                ans.append( self.decoder.id_decode(id+26-vocab_size) )
        return ans
    def detokenize(self, id_list): # id list [26 , 15 , 23 ...]
        import jieba
        jieba.initialize()  # 確保 jieba.dt.FREQ 已載入

        raw = self.id_decode(id_list)

        # 合併連續的 <BYTE_x> token，還原成原本的 utf-8 字元（例如 emoji）
        merged = []
        i = 0
        while i < len(raw):
            item = raw[i]
            if isinstance(item, str) and item.startswith("<BYTE_"):
                byte_buf = []
                while i < len(raw) and isinstance(raw[i], str) and raw[i].startswith("<BYTE_"):
                    byte_buf.append(int(raw[i][6:-1]))
                    i += 1
                try:
                    merged.append(bytes(byte_buf).decode('utf-8'))
                except UnicodeDecodeError:
                    merged.append("�")
                continue
            merged.append(item)
            i += 1

        # 用結巴消歧倉頡同碼字
        result = []
        for item in merged:
            if not isinstance(item, list):
                result.append(item)
                continue

            # 往前抓最近幾個已確定的單一字元當作組詞的上下文
            context = "".join(c for c in result[-4:] if isinstance(c, str) and len(c) == 1)

            best_char, best_freq, found_word = None, -1, False
            for cand in item:
                # 檢查 context 的各種後綴 + cand 是否為 jieba 詞典中的詞
                for start in range(len(context)):
                    word = context[start:] + cand
                    freq = jieba.dt.FREQ.get(word)
                    if freq and freq > best_freq:
                        best_freq, best_char, found_word = freq, cand, True

            if not found_word:
                # 沒有任何候選字能跟上下文組詞，退回比較單字詞頻
                best_char = max(item, key=lambda c: jieba.dt.FREQ.get(c, 0))

            result.append(best_char)

        return "".join(result)

class embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        tok = tokenizer()
        self.CJ_START = tok.vocab['cj_a']
        self.CJ_END = tok.vocab['cj_z']
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.position = nn.Linear(embed_size * 5, embed_size, bias=False)

    def forward(self, x): # x= B , T , 5

        all_emb = self.token_emb(x) # B T 5 E
        first_id = x[:,:,0]
        is_cj = (first_id >= self.CJ_START) & (first_id <= self.CJ_END) # B T

        non_cj_emb = all_emb[:, :, 0 , :] # (B ,T ,E)
        cj_emb = self.position(all_emb.flatten(2))
        if cj_emb.dtype != non_cj_emb.dtype:
            cj_emb = cj_emb.to(non_cj_emb.dtype)

        is_cj = is_cj.unsqueeze(-1)
        return torch.where(is_cj, cj_emb, non_cj_emb)


class Head(nn.Module):
    def __init__(self , head_size ):
        super().__init__()
        self.query = nn.Linear( embed_size, head_size , bias = False )
        self.value = nn.Linear( embed_size, head_size , bias = False )
        self.key = nn.Linear( embed_size, head_size , bias = False )
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # self.dropout = nn.Dropout(dropout)
    def forward( self , x ):
        B,T,C = x.shape
        q =  self.query(x)
        v =  self.value(x)
        k =  self.key(x)
        # w = q @ k.transpose(-2 , -1) * (k.shape[-1]**-0.5)
        # w = w.masked_fill(self.tril[:T , :T] == 0 , float('-inf'))
        # w = F.softmax(w , dim = -1)
        # w = self.dropout(w)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=dropout if self.training else 0.0
        )
        return out # w @ v


class Mutihead(nn.Module):
    def __init__(self , n_head , head_size , embed_size ):
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size
        self.heads = nn.ModuleList([Head( head_size ) for _ in range(n_head)])
        self.proj = nn.Linear( head_size * n_head , embed_size , bias = False )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('_packed_qkv_weight', torch.empty(0), persistent=False)
        self._packed_qkv_versions = None

    def _get_packed_qkv_weight(self):
        if self.training and torch.is_grad_enabled():
            q_weight = torch.cat([head.query.weight for head in self.heads], dim=0)
            k_weight = torch.cat([head.key.weight for head in self.heads], dim=0)
            v_weight = torch.cat([head.value.weight for head in self.heads], dim=0)
            return torch.cat((q_weight, k_weight, v_weight), dim=0)

        current_versions = tuple(
            weight._version
            for head in self.heads
            for weight in (head.query.weight, head.key.weight, head.value.weight)
        )
        first_weight = self.heads[0].query.weight
        needs_refresh = (
            self._packed_qkv_weight.numel() == 0
            or self._packed_qkv_versions != current_versions
            or self._packed_qkv_weight.device != first_weight.device
            or self._packed_qkv_weight.dtype != first_weight.dtype
        )
        if needs_refresh:
            q_weight = torch.cat([head.query.weight for head in self.heads], dim=0)
            k_weight = torch.cat([head.key.weight for head in self.heads], dim=0)
            v_weight = torch.cat([head.value.weight for head in self.heads], dim=0)
            self._packed_qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0).detach()
            self._packed_qkv_versions = current_versions
        return self._packed_qkv_weight

    def forward( self ,x ):
        B, T, _ = x.shape
        qkv = F.linear(x, self._get_packed_qkv_weight())
        q, k, v = qkv.split(self.n_head * self.head_size, dim=-1)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size)
        out = self.proj(out)
        out = self.dropout(out )
        return out


class FF(nn.Module):
    def __init__(self, embed_size ):
        super().__init__()
        self.ff=nn.Sequential(
            nn.Linear(embed_size , embed_size * 4),
            nn.SiLU(),
            nn.Linear(embed_size *4 , embed_size),
            nn.Dropout(dropout),
            )
    def forward(self , x ):
        return self.ff(x)


class layer(nn.Module):
    def __init__(self, n_head , embed_size ):
        super().__init__()
        assert embed_size % n_head == 0, "embed_size 必須能被 n_head 整除"
        head_size = embed_size // n_head
        self.mh = Mutihead(n_head , head_size , embed_size )
        self.ff = FF(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
    def forward( self , x ):
        x = x + self.mh(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class cj_head(nn.Module):
    def __init__(self, emb_layer):
        super().__init__()
        self.emb_layer = emb_layer
        tok = tokenizer()
        codes = tok.all_vocab()  # (13228, 5)
        self.register_buffer('output_codes', codes)
        cj_start = tok.vocab['cj_a']
        cj_end = tok.vocab['cj_z']
        is_cj_output = (codes[:, 0] >= cj_start) & (codes[:, 0] <= cj_end)
        self.register_buffer('non_cj_output_indices', torch.nonzero(~is_cj_output, as_tuple=False).squeeze(1), persistent=False)
        self.register_buffer('cj_output_indices', torch.nonzero(is_cj_output, as_tuple=False).squeeze(1), persistent=False)
        self.register_buffer('non_cj_output_ids', codes[~is_cj_output, 0].to(torch.long), persistent=False)
        self.register_buffer('cj_output_codes', codes[is_cj_output].to(torch.long), persistent=False)
        codes_long = codes.to(torch.long)
        hashes = (
            (codes_long[:, 0] << 36) |
            (codes_long[:, 1] << 27) |
            (codes_long[:, 2] << 18) |
            (codes_long[:, 3] << 9)  |
            codes_long[:, 4]
        ) # needs to be fixed if vocab_size add
        sorted_hashes, sorted_indices = torch.sort(hashes)
        self.register_buffer('sorted_hashes', sorted_hashes)   # (13228,) int64
        self.register_buffer('sorted_indices', sorted_indices) # (13228,) int64
        self.register_buffer('_cached_output_emb', torch.empty(0), persistent=False)
        self._cached_output_versions = None
        # self.tuple_to_id = {tuple(c.tolist()): i for i, c in enumerate(codes)}

    def input_to_output_idx(self, target):
        # target (B, 5) -> output_idx (B, 13228)
        target_long = target.to(torch.long)
        target_hash = (
            (target_long[..., 0] << 36) |
            (target_long[..., 1] << 27) |
            (target_long[..., 2] << 18) |
            (target_long[..., 3] << 9)  |
            target_long[..., 4]
        )
        pos = torch.searchsorted(self.sorted_hashes, target_hash)
        return self.sorted_indices[pos]

    def _build_output_emb(self):
        token_weight = self.emb_layer.token_emb.weight
        output_emb = token_weight.new_empty((self.output_codes.size(0), token_weight.size(1)))

        if self.non_cj_output_ids.numel() > 0:
            non_cj_emb = F.embedding(self.non_cj_output_ids, token_weight)
            output_emb.index_copy_(0, self.non_cj_output_indices, non_cj_emb)

        if self.cj_output_codes.numel() > 0:
            cj_token_emb = F.embedding(self.cj_output_codes, token_weight).flatten(1)
            cj_emb = self.emb_layer.position(cj_token_emb)
            output_emb.index_copy_(0, self.cj_output_indices, cj_emb)

        return output_emb

    def _build_output_emb_for_indices(self, output_indices):
        token_weight = self.emb_layer.token_emb.weight
        codes = self.output_codes.index_select(0, output_indices).to(torch.long)
        first_ids = codes[:, 0]
        cj_start = self.emb_layer.CJ_START
        cj_end = self.emb_layer.CJ_END
        is_cj = (first_ids >= cj_start) & (first_ids <= cj_end)

        non_cj_emb = F.embedding(first_ids, token_weight)
        cj_token_emb = F.embedding(codes, token_weight).flatten(1)
        cj_emb = self.emb_layer.position(cj_token_emb)
        if cj_emb.dtype != non_cj_emb.dtype:
            cj_emb = cj_emb.to(non_cj_emb.dtype)
        return torch.where(is_cj.unsqueeze(-1), cj_emb, non_cj_emb)


    def output_emb(self):
        if self.training and torch.is_grad_enabled():
            return self._build_output_emb()

        token_weight = self.emb_layer.token_emb.weight
        position_weight = self.emb_layer.position.weight
        current_versions = (token_weight._version, position_weight._version)
        needs_refresh = (
            self._cached_output_emb.numel() == 0
            or self._cached_output_versions != current_versions
            or self._cached_output_emb.device != token_weight.device
            or self._cached_output_emb.dtype != token_weight.dtype
        )
        if needs_refresh:
            self._cached_output_emb = self._build_output_emb().detach()
            self._cached_output_versions = current_versions
        return self._cached_output_emb

    def logits(self, hidden):
        return hidden @ self.output_emb().T

    def loss(self, hidden, target):
        target_idx = target.to(torch.long) if target.dim() == 2 else self.input_to_output_idx(target)
        hidden = hidden.flatten(0, 1)
        target_idx = target_idx.reshape(-1)

        if sampled_softmax_negatives <= 0 or sampled_softmax_negatives >= self.output_codes.size(0):
            logits = self.logits(hidden)
            return F.cross_entropy(logits, target_idx)

        negative_idx = torch.randint(
            self.output_codes.size(0),
            (sampled_softmax_negatives,),
            device=target_idx.device,
            dtype=target_idx.dtype,
        )
        sampled_idx, inverse = torch.unique(
            torch.cat((target_idx, negative_idx)),
            sorted=True,
            return_inverse=True,
        )
        target_pos = inverse[:target_idx.numel()]
        sampled_emb = self._build_output_emb_for_indices(sampled_idx)
        logits = hidden @ sampled_emb.T
        return F.cross_entropy(logits, target_pos)

    def forward(self, hidden):
        return self.logits(hidden)


class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.layers = nn.Sequential(*[layer(n_head, embed_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = cj_head(self.embedding)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, target=None):
        B, T, C = x.shape
        emb = self.embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        h = self.ln_f(self.layers(emb + pos_emb))

        loss = None
        if target is not None:
            loss = self.head.loss(h, target)
            logits = self.head(h) if return_training_logits else None
        else:
            logits = self.head(h)

        return logits, loss

@torch.no_grad()
def save_checkpoint( state_dict , save_path ):
    def _save():
        torch.save(state_dict , save_path )
        print(f"saved to {save_path}")
    thread = threading.Thread(target=_save, daemon=True)
    thread.start()


def load_checkpoint(load_path, map_location):
    state_dict = torch.load(load_path, map_location=map_location)
    normalized_state_dict = {}
    for key, value in state_dict.items():
        while key.startswith("_orig_mod.") or key.startswith("module."):
            if key.startswith("_orig_mod."):
                key = key.removeprefix("_orig_mod.")
            if key.startswith("module."):
                key = key.removeprefix("module.")
        normalized_state_dict[key] = value
    return normalized_state_dict


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def output_id_for_token(tok, token_name):
    token_id = tok.vocab.get(token_name)
    if token_id is None:
        return None
    output_id = token_id - 26
    return output_id if output_id >= 0 else None


def apply_generation_filters(logits, tok, generated_ids, repetition_penalty=1.2, allow_byte_tokens=False):
    banned_ids = [
        output_id_for_token(tok, "[PAD]"),
        output_id_for_token(tok, "[BOS]"),
        output_id_for_token(tok, "[UNK]"),
    ]
    for token_id in banned_ids:
        if token_id is not None and token_id < logits.size(-1):
            logits[:, token_id] = float("-inf")

    if not allow_byte_tokens:
        for token_name, token_id in tok.vocab.items():
            if token_name.startswith("<BYTE_"):
                output_id = token_id - 26
                if 0 <= output_id < logits.size(-1):
                    logits[:, output_id] = float("-inf")

    if repetition_penalty != 1.0:
        for token_id in set(generated_ids):
            if 0 <= token_id < logits.size(-1):
                token_logits = logits[:, token_id]
                logits[:, token_id] = torch.where(
                    token_logits > 0,
                    token_logits / repetition_penalty,
                    token_logits * repetition_penalty,
                )

    return logits


def encode_prompt_to_output_ids(prompt_text, tok, head, target_device):
    if not prompt_text:
        return []
    prompt_tokens = tok.tokenize(prompt_text)
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=target_device)
    return head.input_to_output_idx(prompt_tensor).tolist()


@torch.no_grad()
def evaluate_loss(model, data_loader, max_batches):
    if data_loader is None:
        return None
    was_training = model.training
    model.eval()
    losses = []
    try:
        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
            y = y.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16_autocast):
                _, loss = model(x, y)
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            losses.append(loss.detach().float().item())
    finally:
        if was_training:
            model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_sample_text(model, tok, prompt_text, max_tokens):
    was_training = model.training
    base_model = unwrap_model(model)
    model.eval()
    try:
        tok_id_list = [2]
        tok_id_list.extend(encode_prompt_to_output_ids(prompt_text, tok, base_model.head, device))
        all_vocab = tok.all_vocab().tolist()
        eos_id = output_id_for_token(tok, "[EOS]")
        for _ in range(max_tokens):
            tok_list = torch.tensor(
                [all_vocab[tok_id] for tok_id in tok_id_list],
                dtype=torch.long,
                device=device,
            ).view(1, -1, 5)
            next_tok, _ = model(tok_list)
            next_tok = apply_generation_filters(next_tok[:, -1, :], tok, tok_id_list)
            next_tok_id = next_tok.argmax(dim=-1).item()
            if eos_id is not None and next_tok_id == eos_id:
                break
            tok_id_list.append(next_tok_id)
        return tok.detokenize(tok_id_list).removeprefix("[BOS]")
    finally:
        if was_training:
            model.train()


def create_optimizer(model):
    if device == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), lr, fused=True)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), lr)


def maybe_enable_multi_gpu(model):
    if gpu_count <= 1:
        return model
    device_ids = list(range(gpu_count))
    print(f"啟用 DataParallel: GPUs={device_ids}")
    return nn.DataParallel(model, device_ids=device_ids)


def maybe_compile_model(model):
    if not enable_torch_compile:
        if device == "cuda" and gpu_count > 1:
            print("多 GPU 模式下停用 torch.compile，避免與 DataParallel 衝突")
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile 不可用，使用 eager mode")
        return model
    try:
        compiled_model = torch.compile(model, mode=torch_compile_mode)
        print(f"torch.compile 已啟用: mode={torch_compile_mode}")
        return compiled_model
    except Exception as exc:
        print(f"torch.compile 啟用失敗，退回 eager mode: {exc}")
        return model

if __name__=="__main__":
    model = LLM()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量: {total_params:>12,} ({total_params / 1e6:.2f} M)")
    print(f"可訓練參數量: {trainable_params:>12,} ({trainable_params / 1e6:.2f} M)")
    print(f"凍結參數量:{total_params-trainable_params:>12,}")
    for name, param in model.named_parameters():
        if param.requires_grad:
             print(f"{name:<40} | {param.numel():,}")
    head = cj_head( embedding(1,1))
    print(head.input_to_output_idx(torch.tensor([[ 30,  26,  26,  26,  26],
        [  7,  14,  20,  20,  10],
        [ 89,  26,  26,  26,  26],
        [ 99,  26,  26,  26,  26],
        [  1,   7,  13,   5,  26],
        [  7,   0,  15,   8,  26],
        [ 44,  26,  26,  26,  26],
        [ 96,  26,  26,  26,  26],
        [ 18,  12,   7,   0,  26],
        [ 64,  26,  26,  26,  26],
        [255,  26,  26,  26,  26],
        [ 12,   6,   1,  26,  26],
        [ 91,  26,  26,  26,  26],
        [ 56,  26,  26,  26,  26],
        [ 21,   9,   7,  22,  26],
        [ 13,   1,  18,   7,  16],
        [ 45,  26,  26,  26,  26],
        [ 24,  17,  18,  20,  26],
        [ 30,  26,  26,  26,  26],
        [ 24,   2,  10,  26,  26],
        [  7,  14,  12,  12,  13],
        [ 24,  17,  16,  12,   1],
        [  3,   0,   7,  20,  26],
        [ 14,   7,  16,  26,  26],
        [ 42,  26,  26,  26,  26],
        [  7,  16,  15,   7,   7],
        [ 19,  22,   3,  26,  26],
        [ 50,  26,  26,  26,  26],
        [ 24,  19,   0,   9,  26],
        [113,  26,  26,  26,  26],
        [  0,  26,  26,  26,  26],
        [ 13,  23,  20,  26,  26]], dtype=torch.int16))) #676767



    train_ds = CangjieDataset(
        dataset_name="opencsg/chinese-fineweb-edu-v2",
        split="train[:100%]",
        block_size=block_size, 
        cache_path="./cangjie_cached.pt"
    )

    dataset_len = len(train_ds)
    tentative_val_size = min(
        max(batch_size, int(dataset_len * validation_ratio)),
        batch_size * validation_max_batches,
    )
    val_size = tentative_val_size if dataset_len > tentative_val_size else 0
    train_size = dataset_len - val_size
    if val_size > 0:
        train_source = Subset(train_ds, range(train_size))
        val_source = Subset(train_ds, range(train_size, dataset_len))
        print(f"資料切分: train={len(train_source):,} | val={len(val_source):,}")
    else:
        train_source = train_ds
        val_source = None
        print("資料量不足以建立 validation split，將只記錄 training loss")

    train_loader = DataLoader(train_source,
                              batch_size,
                              shuffle=True,
                              collate_fn=collate_cangjie_batch,
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=4)
    val_loader = None
    if val_source is not None:
        val_loader = DataLoader(
            val_source,
            batch_size,
            shuffle=False,
            collate_fn=collate_cangjie_batch,
            num_workers=0,
            pin_memory=True,
        )
    for batch , target in train_loader:
        print("Batch shape:", batch.shape)  # torch.Size([32, 256, 5])
        print("Target shape" , target.shape )  # torch.Size([32, 256])
        break

    state_dict = load_checkpoint("./best_val.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model = maybe_enable_multi_gpu(model)
    model = maybe_compile_model(model)
    optimizer = create_optimizer(model)
    model.train()
    total_steps = epochs * len(train_loader)
    best_val_loss = float("inf")
    best_plateau_metric = float("inf")
    bad_intervals = 0
    lr_scale = 1.0
    for epoch in range(epochs):
        num_batches = len(train_loader)
        running_loss = None
        interval_steps = 0
        interval_start_time = time.perf_counter()
        print(f"epoch{epoch} starts")
        for step,(x, y) in enumerate(train_loader):
            global_step = epoch * num_batches + step
            base_lr = get_lr(global_step, total_steps)
            current_lr = max(plateau_min_lr, base_lr * lr_scale)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            x = x.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
            y = y.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16_autocast):
                logits , loss = model(x , y )
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            loss_for_log = loss.detach()
            running_loss = loss_for_log if running_loss is None else running_loss + loss_for_log
            interval_steps += 1
            is_log_step = (step + 1) % log_interval == 0 or (step + 1) == num_batches
            is_checkpoint_step = (step + 1) % checkpoint_interval == 0 or (step + 1) == num_batches
            if is_log_step:
                elapsed = time.perf_counter() - interval_start_time
                steps_per_sec = interval_steps / elapsed if elapsed > 0 else 0.0
                progress = (step + 1) / num_batches * 100
                avg_loss = (running_loss / interval_steps).item()
                plateau_metric = avg_loss
                plateau_metric_name = "train Loss"
                print(
                    f"epoch [{epoch+1}/{epochs}] | "
                    f"step [{step+1}/{num_batches}] ({progress:.1f}%) | "
                    f"avg Loss: {avg_loss:.4f} | "
                    f"lr: {current_lr:.2e} | "
                    f"steps/s: {steps_per_sec:.2f}"
                )
                val_loss = evaluate_loss(model, val_loader, validation_max_batches)
                if val_loss is not None:
                    print(f"validation Loss: {val_loss:.4f}")
                    plateau_metric = val_loss
                    plateau_metric_name = "validation Loss"
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            {k: v.cpu().clone() for k, v in unwrap_model(model).state_dict().items()},
                            "best_val.pt"
                        )
                        print(f"saved best validation checkpoint: best_val.pt ({best_val_loss:.4f})")
                tok = tokenizer()
                for prompt_text in sample_prompts:
                    sample = generate_sample_text(model, tok, prompt_text, sample_max_tokens)
                    print(f"sample[{prompt_text or '<empty>'}]: {sample}")
                if plateau_metric < best_plateau_metric - plateau_min_delta:
                    best_plateau_metric = plateau_metric
                    bad_intervals = 0
                else:
                    bad_intervals += 1
                    if bad_intervals >= plateau_patience and current_lr > plateau_min_lr:
                        lr_scale *= plateau_factor
                        bad_intervals = 0
                        next_lr = max(plateau_min_lr, base_lr * lr_scale)
                        print(
                            f"reduce lr on plateau: "
                            f"best {plateau_metric_name}: {best_plateau_metric:.4f} | "
                            f"lr_scale: {lr_scale:.4f} | "
                            f"next lr: {next_lr:.2e}"
                        )
                running_loss = None
                interval_steps = 0
                interval_start_time = time.perf_counter()
            if is_checkpoint_step:
                save_checkpoint(
                    {k: v.cpu().clone() for k, v in unwrap_model(model).state_dict().items()},
                    f"cangjie_epoch_{epoch+1}_latest.pt"
                )

    # a=tokenizer()
    # print(a.tokenize("我是abc123🥰："))
