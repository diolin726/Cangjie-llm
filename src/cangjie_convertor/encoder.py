import json
from pathlib import Path
class cj_encoder:
    def __init__(self , vocab_size):
        self.vocab_size = vocab_size
        self.make_vocab()
        with open(Path(__file__).resolve().parent / "cj5.json", "r", encoding="utf-8") as f:
            self.data_map = json.load(f)
        cj_keys = []
        for key in self.data_map:
            cj_keys.append(self.data_map[key])
        cj_keys = list(set(cj_keys))
        self.reversed_cj_key = {key: i for i, key in enumerate(cj_keys)}

    def make_vocab(self):
        self.vocab={}
        self.id_to_vocab=[]
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
            self.id_to_vocab.append(s)
        for i in range(256):
            byte_token = f"<BYTE_{i}>"
            self.vocab[byte_token] = len(self.vocab)
            self.id_to_vocab.append(byte_token)

    def encode(self, s):
        ans=[]
        for token in s:
            if (token not in self.data_map):
                ans.append([token])
            else:
                tmp = ["cj_"+t for t in self.data_map[token] ]
                while len(tmp) < 5:
                    tmp.append("[PAD]")
                ans.append(tmp)
        return ans
    def encode_to_id(self , text): # Warning!! only when input are in token table
        ans = []
        for token in text:
            if token in self.data_map:
                ans.append(  self.reversed_cj_key[self.data_map[token]] + self.vocab_size -26 )
            else:
                ans.append( self.vocab[ token ] -26 )
        return ans

if __name__ == '__main__':
    e = cj_encoder()
    a = e.encode("我是abc123🥰")
    print(a)
    print(len(e.data_map))
