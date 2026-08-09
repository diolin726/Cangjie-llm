import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
import numpy as np
import os
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from cangjie_convertor import cj_encoder


torch.manual_seed(67)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_size = 64
vocab_size = 383 #需要手動調整
batch_size = 32
block_size = 256

class CangjieDataset(Dataset):
    def __init__(self, ds, block_size=256, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            # 快取存在，直接載入（秒級）
            print(f"從快取載入: {cache_path}")
            self.data = torch.load(cache_path, weights_only=True)
            print(f"載入完成: shape={self.data.shape}, 記憶體={self.data.element_size() * self.data.nelement() / 1024**3:.2f} GB")
        else:
            # 首次：批次預處理 Arrow 資料 + 切塊
            print("首次預處理（後續會從快取載入）...")
            batch_sz = 50000
            chunks = []
            for start in range(0, len(ds), batch_sz):
                end = min(start + batch_sz, len(ds))
                batch_ids = ds[start:end]["input_ids"]
                for ids in batch_ids:
                    n = len(ids)
                    if n < block_size:
                        continue
                    arr = np.array(ids, dtype=np.int16)
                    num_blocks = (n - block_size) // block_size + 1
                    for i in range(num_blocks):
                        s = i * block_size
                        chunks.append(arr[s : s + block_size])
                print(f"  已處理 {end}/{len(ds)} 筆，共 {len(chunks)} chunks")

            self.data = torch.from_numpy(np.stack(chunks))  # int16 tensor
            # shape: (num_samples, block_size, 5)
            print(f"預處理完成: shape={self.data.shape}, 記憶體={self.data.element_size() * self.data.nelement() / 1024**3:.2f} GB")

            if cache_path:
                torch.save(self.data, cache_path)
                print(f"已儲存快取: {cache_path}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].to(torch.long)


class tokenizer():

    def __init__(self):
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
        for s in special_tokens + cangjie_symbols + ascii_chars:
            self.vocab[s] = len(self.vocab)
        for i in range(256):
            byte_token = f"<BYTE_{i}>"
            self.vocab[byte_token] = len(self.vocab)

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
    # cj_a=4 ~ cj_z=29
    CJ_START = 4
    CJ_END = 29

    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.position = nn.linear( embed_size * 5 , embed_size ) # can try other structure

    def forward(self, x):
        # (B, T, 5)
        B, T, C = x.shape

        all_emb = self.token_emb(x)              # (B, T, 5, E)

        first_id = x[:, :, 0]                     # (B, T)
        is_cj = (first_id >= self.CJ_START) & (first_id <= self.CJ_END)  # (B, T)

        non_cj_emb = all_emb[:, :, 0, :]           # (B, T, E)

        cj_emb = self.position( all_emb.view(B,T,-1) )

        is_cj = is_cj.unsqueeze(-1)                 # (B, T, 1)
        output = torch.where(is_cj, cj_emb, non_cj_emb)  # (B, T, E)

        return output
'''
#download ppt_pretrain.json from yuhuanstudio/PTT-pretrain-zhtw on huggingface
abc = tokenizer()
dataset = load_dataset("json", data_files="ppt_pretrain.json", split="train")
def process_fn(example):
    text = example["text"] if example["text"] else ""
    return {"input_ids": abc.tokenize(text)}
print("開始平行 Tokenize...")
tokenized_ds = dataset.map(
    process_fn,
    remove_columns=["text"],  # 轉完即刪除原始文字，節省空間
    num_proc=8,               # 根據你的 CPU 核心數設定（如 8 或 16）
    desc="Processing PTT Dataset"
)

# 5. 將處理完的結果直接「儲存在硬碟」（存成 Arrow 格式）
# 這樣你下次開機訓練時，不用重新 tokenize，1 秒鐘就能載入！
tokenized_ds.save_to_disk("./ptt_cangjie_arrow")
print("處理完成並已儲存至硬碟！")
'''
if __name__=="__main__":

    tokenized_ds = load_from_disk("./ptt_cangjie_arrow")
    train_ds = CangjieDataset(tokenized_ds, block_size, cache_path="./ptt_cangjie_cached.pt")
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    for batch in train_loader:
        print("Batch shape:", batch.shape)  # torch.Size([32, 256, 5])
        break

    a=tokenizer()
    print(a.tokenize("我是abc123🥰："))
