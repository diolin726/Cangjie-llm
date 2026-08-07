import json
import os
import urllib.request
from typing import List

# 倉頡字母映射表 (A-Z -> 倉頡字母)
CJ_RADICAL_MAP = {
    'a': '日', 'b': '月', 'c': '金', 'd': '木', 'e': '水', 'f': '火', 'g': '土',
    'h': '竹', 'i': '戈', 'j': '十', 'k': '大', 'l': '中', 'm': '一', 'n': '弓',
    'o': '人', 'p': '心', 'q': '手', 'r': '口', 's': '尸', 't': '廿', 'u': '山',
    'v': '女', 'w': '田', 'x': '難', 'y': '卜', 'z': '重'
}

# 倉頡 26 個字母 + 1 個 [PAD_CJ]，共 27 個 Token
PAD_CJ_TOKEN = '[PAD_CJ]'
ALL_CJ_TOKENS = list(CJ_RADICAL_MAP.values()) + [PAD_CJ_TOKEN]  # 剛好 27 個 Token

DICT_FILE = os.path.join(os.path.dirname(__file__), 'cangjie_dict.json')


def _download_and_build_dict() -> dict:
    """如果本地字典不存在，從網路下載標準倉頡五代/三代碼表並建立字典檔。"""
    print("正在初始化倉頡字庫，請稍候...")
    cj5_url = 'https://raw.githubusercontent.com/fcitx/fcitx-table-extra/master/tables/cangjie5.txt'
    cj3_url = 'https://raw.githubusercontent.com/fcitx/fcitx-table-extra/master/tables/cangjie3.txt'

    def load_table(url):
        with urllib.request.urlopen(url) as resp:
            lines = resp.read().decode('utf-8').splitlines()
        mapping = {}
        in_data = False
        for line in lines:
            if line == '[數據]':
                in_data = True
                continue
            if in_data:
                parts = line.strip().split()
                if len(parts) == 2:
                    code, char = parts
                    if not code.startswith('&') and char not in mapping:
                        mapping[char] = code
        return mapping

    try:
        m3 = load_table(cj3_url)
        m5 = load_table(cj5_url)
        combined = {**m3, **m5}
        with open(DICT_FILE, 'w', encoding='utf-8') as f:
            json.dump(combined, f, ensure_ascii=False)
        print(f"字庫建置完成，共收錄 {len(combined)} 個字。")
        return combined
    except Exception as e:
        raise RuntimeError(f"下載倉頡字典失敗: {e}")

def load_cangjie_dict() -> dict:
    """載入倉頡字典"""
    if not os.path.exists(DICT_FILE):
        return _download_and_build_dict()
    with open(DICT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

# 全局載入字庫
_CJ_DICT = None


def get_char_cangjie_tokens(char: str, max_len: int = 5) -> List[str]:
    """
    將單個漢字拆解為長度固定為 5 的倉頡字根列表，不足者以 '[PAD_CJ]' 補齊。
    例如:
      "明" -> ['日', '月', '[PAD_CJ]', '[PAD_CJ]', '[PAD_CJ]']
      "晶" -> ['日', '日', '日', '[PAD_CJ]', '[PAD_CJ]']
    """
    global _CJ_DICT
    if _CJ_DICT is None:
        _CJ_DICT = load_cangjie_dict()

    if char in _CJ_DICT:
        code = _CJ_DICT[char]
        radicals = [CJ_RADICAL_MAP.get(letter.lower(), letter) for letter in code[:max_len]]
    else:
        radicals = []

    while len(radicals) < max_len:
        radicals.append(PAD_CJ_TOKEN)

    return radicals


def text_to_cangjie(text: str, ignore_unknown: bool = False) -> List[str]:

    """
    將簡體/繁體中文文本轉換為倉頡拆碼列表，每個字結尾加上 '0'。

    :param text: 輸入的繁簡體字串，例如 "明林"
    :param ignore_unknown: 若為 True，則忽略無法轉換的非漢字（如標點或英文）
    :return: 倉頡拆碼列表，例如 ['日', '月', '0', '木', '木', '0']
    """
    global _CJ_DICT
    if _CJ_DICT is None:
        _CJ_DICT = load_cangjie_dict()

    result = []
    for char in text:
        if char in _CJ_DICT:
            code = _CJ_DICT[char]
            for letter in code:
                # 轉為對應的倉頡字母
                radical = CJ_RADICAL_MAP.get(letter.lower(), letter)
                result.append(radical)
            result.append('0')
        elif not ignore_unknown:
            # 如果非字典內的字或符號，保留原字並加 '0'
            result.append(char)
            result.append('0')

    return result

if __name__ == '__main__':
    # 測試範例
    sample_text = "明林"
    res = text_to_cangjie(sample_text)
    print(f"輸入: {sample_text}")
    print(f"輸出: {res}")

    print("\n更多測試 (簡繁體混排):")
    test_cases = ["唱晶","明林", "繁體字", "简体字", "倉頡輸入法"]
    for t in test_cases:
        print(f"'{t}' -> {text_to_cangjie(t)}")
