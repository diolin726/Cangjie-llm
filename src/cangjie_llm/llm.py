import torch
import torch.nn as nn
import torch.nn.functional as F

from cangjie_convertor import cj_encoder

embed_lenth = 64


class tokenizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.make_vocab()
        self.cj_encoder=cj_encoder()

    @torch.no_grad()
    def make_vocab(self):
        self.vocab={}
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        cangjie_symbols = [
                'cj_a', 'cj_b', 'cj_c', 'cj_d', 'cj_e', 'cj_f',
                'cj_g', 'cj_h', 'cj_i', 'cj_j','cj_k', 'cj_l',
                'cj_m', 'cj_n', 'cj_o', 'cj_p', 'cj_q', 'cj_r',
                'cj_s', 'cj_t','cj_u', 'cj_v', 'cj_w', 'cj_x', 'cj_y', 'cj_z',
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
            return [[ self.vocab[token_list[0]] ]]

        try:
            utf8_bytes = token_list[0].encode('utf-8')
            return [ [ self.vocab[f"<BYTE_{b}>"] ] for b in utf8_bytes]
        except Exception:
            return [[self.vocab["[UNK]"]]]

    def tokenize(self, s ):
        s = self.cj_encoder.encode(s)
        ans=[]
        for token_list in s :
            id_list = self.tokenlist_to_id(token_list)
            ans = ans + id_list
        return ans


if __name__=="__main__":
    a=tokenizer()
    print(a.tokenize("我是abc123🥰"))
