from ._shared import load_cj_assets


class cj_decoder:
    def __init__(self):
        self.data_map, self.cj_keys, _, self.cj_decodemap, _ = load_cj_assets()

    def decode(self, s): # input a char [ 'cj_a', 'cj_b', '[PAD]' , '[PAD]' , '[PAD]' ]
        key_parts = []
        for token in s:
            token = token[3]
            if 'a'<= token <= 'z':
                key_parts.append(token)
        return self.cj_decodemap["".join(key_parts)]

    def id_decode(self, idx):
        return self.cj_decodemap[self.cj_keys[idx]]


if __name__ == "__main__":
    a = cj_decoder()
    print(len(a.cj_keys))
    print (a.decode([ 'cj_a', 'cj_f', '[PAD]' , '[PAD]' , '[PAD]' ]) )

