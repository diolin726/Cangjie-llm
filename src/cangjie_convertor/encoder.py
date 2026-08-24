from ._shared import load_cj_assets


class cj_encoder:
    def __init__(self , vocab_size):
        self.vocab_size = vocab_size
        self.make_vocab()
        self.data_map, _, self.reversed_cj_key, _, self.encoded_tokens = load_cj_assets()
        self.pad_id = self.vocab["[PAD]"]
        self.unk_row = (
            self.vocab["[UNK]"],
            self.pad_id,
            self.pad_id,
            self.pad_id,
            self.pad_id,
        )
        self.single_token_rows = {
            token: (token_id, self.pad_id, self.pad_id, self.pad_id, self.pad_id)
            for token, token_id in self.vocab.items()
        }
        self.encoded_token_rows = {
            char: tuple(self.vocab[token] for token in tokens)
            for char, tokens in self.encoded_tokens.items()
        }
        self.byte_rows_cache = {}
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
        import json
        from pathlib import Path
        with open(Path(__file__).resolve().parent /".json", "r", encoding="utf-8") as f:
            chinese_vocab_data = json.load(f)
        for token in chinese_vocab_data:
            self.vocab[token] = len(self.vocab)
            self.id_to_vocab.append(token)

    def encode(self, s):
        encoded_tokens = self.encoded_tokens
        return [encoded_tokens.get(token, [token]) for token in s]

    def encode_text_to_rows(self, s):
        encoded_token_rows = self.encoded_token_rows
        single_token_rows = self.single_token_rows
        byte_rows_cache = self.byte_rows_cache
        unk_row = self.unk_row
        pad_id = self.pad_id
        vocab = self.vocab
        rows = []
        append = rows.append
        extend = rows.extend

        for token in s:
            encoded = encoded_token_rows.get(token)
            if encoded is not None:
                append(encoded)
                continue

            single = single_token_rows.get(token)
            if single is not None:
                append(single)
                continue

            cached = byte_rows_cache.get(token)
            if cached is None:
                try:
                    cached = tuple(
                        (vocab[f"<BYTE_{byte}>"], pad_id, pad_id, pad_id, pad_id)
                        for byte in token.encode("utf-8")
                    )
                except Exception:
                    cached = (unk_row,)
                byte_rows_cache[token] = cached
            extend(cached)

        return rows

    def encode_to_id(self , text): # Warning!! only when input are in token table
        char_to_output_id = self.char_to_output_id
        vocab = self.vocab
        return [char_to_output_id.get(token, vocab[token] - 26) for token in text]

if __name__ == '__main__':
    e = cj_encoder()
    a = e.encode("我是abc123🥰")
    print(a)
    print(len(e.data_map))
