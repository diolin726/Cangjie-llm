import json
from pathlib import Path
class cj_decoder:
    def __init__(self):
        with open(Path(__file__).resolve().parent / "cj5.json", "r", encoding="utf-8") as f:
            self.data_map = json.load(f)

        self.cj_keys = []
        self.cj_decodemap={}
        for key in self.data_map:
            self.cj_keys.append(self.data_map[key])
            if self.data_map[key] not in self.cj_decodemap:
                self.cj_decodemap[self.data_map[key]] = []
            self.cj_decodemap[self.data_map[key]].append(key)
        #print(self.cj_decodemap)

        self.cj_keys = list(set(self.cj_keys))

    def decode(self, s): # input a char [ 'cj_a', 'cj_b', '[PAD]' , '[PAD]' , '[PAD]' ]
        key = ''
        for t in s:
            t = t[3]
            if 'a'<= t <= 'z':
                key = key + t
            #print(key)

        return self.cj_decodemap[key]

    def id_decode(self, idx):
        return self.cj_decodemap[self.cj_keys[idx]]


if __name__ == "__main__":
    a = cj_decoder()
    print(len(a.cj_keys))
    print (a.decode([ 'cj_a', 'cj_f', '[PAD]' , '[PAD]' , '[PAD]' ]) )

