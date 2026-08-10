import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from cangjie_convertor import cj_encoder , cj_decoder


torch.manual_seed(67) #676767
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print( f"using {device}" )
embed_size = 256
vocab_size = 383 #需要手動調整
batch_size = 32
block_size = 256
n_head = 8
n_layer = 8
lr = 1e-4
epochs = 1
log_interval = 1000

class CangjieDataset(Dataset): # this part is by ai, im sorry but im trash
    def __init__(self, json_path="ppt_pretrain.json", block_size=256, cache_path=None):
        self.block_size = block_size
        if cache_path and os.path.exists(cache_path):
            print(f"從快取載入: {cache_path}")
            self.data = torch.load(cache_path, weights_only=True) #.to(torch.long)
            print(f"載入完成: shape={self.data.shape}, 記憶體={self.data.element_size() * self.data.nelement() / 1024**3:.2f} GB")
        else:
            print("首次預處理（後續會從快取載入）...")
            from datasets import load_dataset
            tok = tokenizer()
            BOS = [tok.vocab["[BOS]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"]]
            EOS = [tok.vocab["[EOS]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"]]

            ds = load_dataset("json", data_files=json_path, split="train")
            all_tokens = []
            for i, row in enumerate(ds):
                text = row.get("text") or ""
                ids = tok.tokenize(text)  # List of 5-tuples
                all_tokens.append(BOS)
                all_tokens.extend(ids)
                all_tokens.append(EOS)
                if (i + 1) % 10000 == 0:
                    print(f"  已處理 {i+1}/{len(ds)} 篇，共 {len(all_tokens)} tokens")

            # 轉成 (N, 5) 的 int16 tensor
            self.data = torch.tensor(all_tokens, dtype=torch.int16)
            print(f"預處理完成: shape={self.data.shape}, 記憶體={self.data.element_size() * self.data.nelement() / 1024**3:.2f} GB")
            if cache_path:
                torch.save(self.data, cache_path)
                print(f"已儲存快取: {cache_path}")
        self.data=self.data.to(torch.long)
#        self.data.share_memory_()
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x: 滑動窗口 (block_size, 5)
        x = self.data[idx : idx + self.block_size]
        # target: 緊接在窗口後面的下一個 token (5,)
        target = self.data[idx + self.block_size]
        return x, target


class tokenizer():
    #[TODO] add jieba to decode
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

    def forward( self , x ):
        B,T,C = x.shape
        q =  self.query(x)
        v =  self.value(x)
        k =  self.key(x)
        w = q @ k.transpose(-2 , -1) * (k.shape[-1]**-0.5)
        w = w.masked_fill(self.tril[:T , :T] == 0 , float('-inf'))
        w = F.softmax(w , dim = -1)
        return w @ v


class Mutihead(nn.Module):
    def __init__(self , n_head , head_size , embed_size ):
        super().__init__()
        self.heads = nn.ModuleList([Head( head_size ) for _ in range(n_head)])
        self.proj = nn.Linear( head_size * n_head , embed_size )

    def forward( self ,x ):
        out = torch.cat([ h(x) for h in self.heads ], dim=-1)
        out = self.proj(out)
        return out


class FF(nn.Module):
    def __init__(self, embed_size ):
        super().__init__()
        self.ff=nn.Sequential(
            nn.Linear(embed_size , embed_size * 4),
            nn.SiLU(),
            nn.Linear(embed_size *4 , embed_size)
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
        self.tuple_to_id = {tuple(c.tolist()): i for i, c in enumerate(codes)}

    def input_to_output_idx(self, target):
        # target (B, 5) -> output_idx (B, 13228)
        device = target.device
        flat = target.tolist()
        indices = [self.tuple_to_id[tuple(t)] for t in flat]
        return torch.tensor(indices, device=device)

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
        h = h[:,-1,:] # (B , E)
        logits = self.head(h)  # (B, 13228)

        loss = None
        if target is not None:
            target_idx = self.head.input_to_output_idx(target) # [B ]
            loss = F.cross_entropy(logits, target_idx)

        return logits, loss

if __name__=="__main__":

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


    train_ds = CangjieDataset( block_size=block_size, cache_path="./ptt_cangjie_cached.pt")

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
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
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
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
    torch.save(model.state_dict(), "cangjie.pt")

    # a=tokenizer()
    # print(a.tokenize("我是abc123🥰："))
