import json
class cj_encoder:
    def __init__(self):
        with open("../cangjie_convertor/cj5.json", "r", encoding="utf-8") as f:
            self.data_map = json.load(f)

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

if __name__ == '__main__':
    e = cj_encoder()
    a = e.encode("我是abc123🥰")
    print(a)
