import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from cangjie_convertor import cj_encoder , cj_decoder
from torch.amp import autocast

torch.manual_seed(67) #676767
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print( f"using {device}" )

dropout=0.1
embed_size = 384
vocab_size = 383 #需要手動調整
batch_size = 64
block_size = 256
n_head = 12
n_layer = 12
lr = 3e-4
epochs = 100
log_interval = 1000

class CangjieDataset(Dataset): # this part is by ai, im sorry but im trash
    def __init__(self, json_path=None, dataset_name=None , dataset_dir=None , block_size=256, cache_path=None):
        self.block_size = block_size
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
                    data_dir=dataset_dir, split="train"
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
         #self.data=self.data.to(torch.long)
#        self.data.share_memory_()
    def __len__(self):
        return (len(self.data) - self.block_size ) * 2 // self.block_size

    def __getitem__(self, idx):
        idx = idx * self.block_size // 2
        x = self.data[idx : idx + self.block_size]
        target = self.data[idx + 1: idx + self.block_size]
        return x, target


class tokenizer():
    #[TODO] add jieba to decode
    #英文就直接轉,id 0 ~ vocab_size -26 -1
    #if decoder.id_decode() returns a list len > 1
    #用結巴確認前面幾個字加目前的候選字是不是一個詞,找詞頻最高的輸出,如果都不是一個字就輸出字本人詞頻最高的
    def __init__(self):
        self.decoder=cj_decoder()
        self.make_vocab()
        self.cj_encoder=cj_encoder()

    def make_vocab(self):
        self.vocab={}
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        cangjie_symbols = [
                'cj_a', 'cj_b', 'cj_c', 'cj_d', 'cj_e', 'cj_f',
                'cj_g', 'cj_h', 'cj_i', 'cj_j', 'cj_k', 'cj_l',
                'cj_m', 'cj_n', 'cj_o', 'cj_p', 'cj_q', 'cj_r',
                'cj_s', 'cj_t', 'cj_u', 'cj_v', 'cj_w', 'cj_x',
                'cj_y', 'cj_z',
                ]
        ascii_chars = [chr(i) for i in range(32, 127)] + ['\n', '\t']
        for s in cangjie_symbols + special_tokens  + ascii_chars:
            self.vocab[s] = len(self.vocab)
        for i in range(256):
            byte_token = f"<BYTE_{i}>"
            self.vocab[byte_token] = len(self.vocab)

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
    def id_decode(self, ids):
        """
        ids: 單一 token，長度 5 的 id list/tuple/tensor（tokenize() 輸出的其中一筆）
        回傳：
          - 特殊符號 [PAD]/[BOS]/[EOS] -> "" （不佔輸出文字）
          - [UNK] -> "�"
          - <BYTE_x> -> 原樣回傳字串 "<BYTE_x>"，交給 detokenize() 合併還原成 utf-8 字元
          - 一般 ASCII / 英文 -> 該字元本身 (str)
          - 倉頡碼只對應 1 個字 -> 該字 (str)
          - 倉頡碼對應多個同碼字 -> list[str]，交給 detokenize() 用 jieba 消歧
        """
        if not hasattr(self, 'inv_vocab'):
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

        first_tok = self.inv_vocab[int(ids[0])]

        # if first_tok in ("[PAD]", "[BOS]", "[EOS]"): #之後要做正常輸出再改就好
        #     return ""
        # if first_tok == "[UNK]":
        #     return "�"
        if not first_tok.startswith("cj_"):
            # 英文/ASCII: id 直接對應字元本身，不需查表轉換
            return first_tok

        # 倉頡碼 token: 把 5 個 id 還原成碼字串，例如 [cj_a, cj_b, PAD, PAD, PAD] -> "ab"
        code = ""
        for tid in ids:
            name = self.inv_vocab[int(tid)]
            if name == "[PAD]":
                break
            code += name[3:]  # 去掉 "cj_" 前綴

        candidates = self.decoder.id_decode(code)  # 假設回傳同碼候選字 list[str]，需依實際 API 調整
        if not candidates:
            return "�"
        if len(candidates) == 1:
            return candidates[0]
        return list(candidates)  # 多個同碼字，留給 detokenize() 消歧

    def detokenize(self, id_list):
        """
        id_list: tokenize() 回傳的格式，一個 list，每個元素是長度 5 的 id list/tensor
        用 jieba 對倉頡同碼字做消歧：
          - 先看前面幾個已確定的字 + 目前候選字，是否能組成 jieba 詞典中的詞，
            取詞頻最高者
          - 都組不成詞的話，退回取候選字中單字詞頻最高者
        """
        import jieba
        jieba.initialize()  # 確保 jieba.dt.FREQ 已載入

        raw = [self.id_decode(ids) for ids in id_list]

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

        all_emb = self.token_emb(x)                          # B T 5 E
        first_id = x[:,:,0]
        is_cj = (first_id >= self.CJ_START) & (first_id <= self.CJ_END) # B T

        non_cj_emb = all_emb[:, :, 0 , :]                             # (B ,T ,E)
        cj_emb = self.position( all_emb.flatten(2) )#(B ,T, E)

        is_cj = is_cj.unsqueeze(-1)
        output = torch.where(is_cj, cj_emb, non_cj_emb)
        return output


class Head(nn.Module):
    def __init__(self , head_size ):
        super().__init__()
        self.query = nn.Linear( embed_size, head_size , bias = False )
        self.value = nn.Linear( embed_size, head_size , bias = False )
        self.key = nn.Linear( embed_size, head_size , bias = False )
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward( self , x ):
        B,T,C = x.shape
        q =  self.query(x)
        v =  self.value(x)
        k =  self.key(x)
        w = q @ k.transpose(-2 , -1) * (k.shape[-1]**-0.5)
        w = w.masked_fill(self.tril[:T , :T] == 0 , float('-inf'))
        w = F.softmax(w , dim = -1)
        w = self.dropout(w)
        return w @ v


class Mutihead(nn.Module):
    def __init__(self , n_head , head_size , embed_size ):
        super().__init__()
        self.heads = nn.ModuleList([Head( head_size ) for _ in range(n_head)])
        self.proj = nn.Linear( head_size * n_head , embed_size )
        self.dropout = nn.Dropout(dropout)

    def forward( self ,x ):
        out = torch.cat([ h(x) for h in self.heads ], dim=-1)  #[TODO] gemini says use F.scaled_dot_product_attention will be faster
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

    def forward(self, hidden):
        # hidden: (B , E) -> logits: (B, 13228)
        output_emb = self.emb_layer(self.output_codes.unsqueeze(1)).squeeze(1) # [13228 , E]
        return hidden @ output_emb.T


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
        # h = h[:,-1,:] # (B , E)
        logits = self.head(h)  # (B, 13228)

        loss = None
        if target is not None:
            target_idx = self.head.input_to_output_idx(target) # [B ]
            loss = F.cross_entropy(logits.view(-1, 13228), target_idx.view(-1))

        return logits, loss

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


    train_ds = CangjieDataset( dataset_name="zaibd/wikipedia-pretrain-zh-tw" ,dataset_dir="2605", block_size=block_size, cache_path="./cangjie_cached.pt")

    train_loader = DataLoader(train_ds,
                              batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=4)
    for batch , target in train_loader:
        print("Batch shape:", batch.shape)  # torch.Size([32, 256, 5])
        print("Target shape" , target.shape )
        break
    model = LLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    model.train()
    for epoch in range(epochs):
        num_batches = len(train_loader)
        print(f"epoch{epoch} starts")
        for step,(x, y) in enumerate(train_loader):
            x = x.to(torch.long).to(device)
            y = y.to(torch.long).to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits , loss = model(x , y )

            loss.backward()
            optimizer.step()
            if (step + 1) % log_interval == 0 or (step + 1) == num_batches:
                progress = (step + 1) / num_batches * 100
                print(
                    f"epoch [{epoch+1}/{epochs}] | "
                    f"step [{step+1}/{num_batches}] ({progress:.1f}%) | "
                    f"current Loss: {loss.item():.4f}"
                )
        torch.save(model.state_dict(), f"cangjie_epoch_{epoch}.pt")

    # a=tokenizer()
    # print(a.tokenize("我是abc123🥰："))
