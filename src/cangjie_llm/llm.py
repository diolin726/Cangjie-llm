import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
import numpy as np
import os
import math
import time
from torch.utils.data import DataLoader, Dataset
from cangjie_convertor import cj_encoder , cj_decoder
from torch.amp import autocast
import threading

torch.manual_seed(67) #676767

# ===== 可調參數 =====
dropout = 0.1
embed_size = 384
vocab_size = 383 #需要手動調整
batch_size = 192
block_size = 256
n_head = 12
n_layer = 12
lr = 3e-4
min_lr = 3e-5
warmup_steps = 1000
plateau_patience = 3
plateau_factor = 0.5
plateau_min_delta = 0.003
plateau_min_lr = 1e-5
epochs = 1
log_interval = 1000
checkpoint_interval = 1000
torch_compile_mode = "default"
return_training_logits = False
sampled_softmax_negatives = 2048

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

class CangjieDataset(Dataset): # this part is by ai, im sorry but im trash
    def __init__(self, json_path=None, dataset_name=None , dataset_dir=None , data_files=None , block_size=256, cache_path=None):
        self.block_size = block_size
        self.target_cache_path = self._get_target_cache_path(cache_path)
        if cache_path and os.path.exists(cache_path):
            print(f"從快取載入: {cache_path}")
            self.data = torch.load(cache_path, weights_only=True) #.to(torch.long)
            print(f"載入完成: shape={self.data.shape}, 記憶體={self.data.element_size() * self.data.nelement() / 1024**3:.2f} GB")
        else:
            print("首次預處理（後續會從快取載入）...")
            from datasets import load_dataset
            import opencc
            stotconverter = opencc.OpenCC('s2twp')
            tok = tokenizer()
            BOS = [tok.vocab["[BOS]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"]]
            EOS = [tok.vocab["[EOS]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"]]

            if json_path is not None :
                ds = load_dataset("json", data_files=json_path, split="train")
            else:
                ds=load_dataset(
                    dataset_name ,
                    data_dir=dataset_dir,
                    data_files=data_files,
                    split="train",
                    )
            chunk_size=10000000
            chunk_tokens = []
            token_count=0
            tensor_list=[]
            print(ds[0])
            for i, row in enumerate(ds):
                if isinstance(row , dict ) :
                    text = row.get("text") or ""
                else:
                    text = row
                text = stotconverter.convert(text)
                ids = tok.tokenize(text)  # List of 5-tuples
                chunk_tokens.append(BOS)
                chunk_tokens.extend(ids)
                chunk_tokens.append(EOS)
                if (i + 1) % 10000 == 0:
                    print(f"  已處理 {i+1}/{len(ds)} 篇，共 {len(chunk_tokens) + token_count} tokens")
                if( len(chunk_tokens) >= chunk_size ):
                    tensor_list.append( torch.tensor(chunk_tokens , dtype=torch.int16))
                    token_count += len(chunk_tokens)
                    chunk_tokens=[]
            if(chunk_tokens):
                tensor_list.append(torch.tensor(chunk_tokens , dtype=torch.int16))
                chunk_tokens=[]

            # 轉成 (N, 5) 的 int16 tensor
            self.data = torch.cat(tensor_list , dim=0)
            print(f"預處理完成: shape={self.data.shape}, 記憶體={self.data.element_size() * self.data.nelement() / 1024**3:.2f} GB")
            if cache_path:
                torch.save(self.data, cache_path)
                print(f"已儲存快取: {cache_path}")

        if self.target_cache_path and os.path.exists(self.target_cache_path):
            print(f"從快取載入 target ids: {self.target_cache_path}")
            self.target_ids = torch.load(self.target_cache_path, weights_only=True)
            print(
                f"target ids 載入完成: shape={self.target_ids.shape}, "
                f"記憶體={self.target_ids.element_size() * self.target_ids.nelement() / 1024**3:.2f} GB"
            )
        else:
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
                print(f"已儲存 target id 快取: {self.target_cache_path}")
        #self.data=self.data.to(torch.long)
        #self.data.share_memory_()

    @staticmethod
    def _get_target_cache_path(cache_path):
        if cache_path is None:
            return None
        root, ext = os.path.splitext(cache_path)
        return f"{root}_target_ids{ext or '.pt'}"

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
        s = self.cj_encoder.encode( unicodedata.normalize('NFKC',s).replace('\u3000', ' ') )
        ans=[]
        for token_list in s :
            id_list = self.tokenlist_to_id(token_list)
            ans.extend(id_list)
        return ans
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

        output_emb = F.embedding(first_ids, token_weight)
        if is_cj.any():
            cj_token_emb = F.embedding(codes[is_cj], token_weight).flatten(1)
            cj_emb = self.emb_layer.position(cj_token_emb)
            output_emb = output_emb.index_copy(0, torch.nonzero(is_cj, as_tuple=False).squeeze(1), cj_emb)
        return output_emb


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



    train_ds = CangjieDataset( dataset_name="opencsg/chinese-fineweb-edu" ,data_files=["cci2/00000*", "cci2/00001*", "cci2/00002*", "cci2/00003*", "cci2/00004*"] , block_size=block_size, cache_path="./cangjie_cached.pt")

    train_loader = DataLoader(train_ds,
                              batch_size,
                              shuffle=True,
                              collate_fn=collate_cangjie_batch,
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=4)
    for batch , target in train_loader:
        print("Batch shape:", batch.shape)  # torch.Size([32, 256, 5])
        print("Target shape" , target.shape )  # torch.Size([32, 256])
        break

    state_dict = load_checkpoint("./cangjie.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model = maybe_enable_multi_gpu(model)
    model = maybe_compile_model(model)
    optimizer = create_optimizer(model)
    model.train()
    total_steps = epochs * len(train_loader)
    best_loss = float("inf")
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
                print(
                    f"epoch [{epoch+1}/{epochs}] | "
                    f"step [{step+1}/{num_batches}] ({progress:.1f}%) | "
                    f"avg Loss: {avg_loss:.4f} | "
                    f"lr: {current_lr:.2e} | "
                    f"steps/s: {steps_per_sec:.2f}"
                )
                if avg_loss < best_loss - plateau_min_delta:
                    best_loss = avg_loss
                    bad_intervals = 0
                else:
                    bad_intervals += 1
                    if bad_intervals >= plateau_patience and current_lr > plateau_min_lr:
                        lr_scale *= plateau_factor
                        bad_intervals = 0
                        next_lr = max(plateau_min_lr, base_lr * lr_scale)
                        print(
                            f"reduce lr on plateau: "
                            f"best Loss: {best_loss:.4f} | "
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
