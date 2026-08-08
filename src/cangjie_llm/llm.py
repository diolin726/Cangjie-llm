import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from cangjie_convertor import cj_encoder


torch.manual_seed(67)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_size = 64
vocab_size = 383 #須要手動調整
batch_size = 32
block_size = 10


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

'''
class embedding(nn.Module):
    def __init__(self ):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size , embed_size)

    def forward( self , token_list ):
        emb=torch.zero()
        for token in token_list:
            for t in token:

'''

'''
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
    a=tokenizer()
    print(dataset)
    print(a.tokenize("我是abc123🥰："))
