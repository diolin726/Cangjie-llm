from ._shared import load_cj_assets


class cj_encoder:
    def __init__(self , vocab_size):
        self.vocab_size = vocab_size
        self.make_vocab()
        self.data_map, _, self.reversed_cj_key, _, self.encoded_tokens = load_cj_assets()
        self.char_to_output_id = {
            char: self.reversed_cj_key[key] + self.vocab_size - 26
            for char, key in self.data_map.items()
        }

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
        encoded_tokens = self.encoded_tokens
        return [encoded_tokens.get(token, [token]) for token in s]

    def encode_to_id(self , text): # Warning!! only when input are in token table
        char_to_output_id = self.char_to_output_id
        vocab = self.vocab
        return [char_to_output_id.get(token, vocab[token] - 26) for token in text]

if __name__ == '__main__':
    e = cj_encoder()
    a = e.encode("我是abc123🥰")
    print(a)
    print(len(e.data_map))
