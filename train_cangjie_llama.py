import argparse
import itertools
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

from cangjie_tokenizer import CangjieLlamaTokenizer, TOTAL_VOCAB_SIZE, BASE_VOCAB_SIZE, PAD_CJ_ID
from modeling_cangjie_llama import CangjieTextGenerationModel

def get_interleaved_dataset_stream(cn_ratio=0.5):
    """
    從 HuggingFace 加載中文與英文 FineWeb-Edu 數據集 (Streaming 模式)，並交替產生文本。
    - 中文數據集: mrhuangtalk/fineweb-edu-chinese-shuffle
    - 英文數據集: Josephgflowers/Par-Four-Fineweb-Edu-Fortified
    """
    print("正在連接與串流數據集...")
    print("  - 中文數據集: mrhuangtalk/fineweb-edu-chinese-shuffle")
    print("  - 英文數據集: Josephgflowers/Par-Four-Fineweb-Edu-Fortified")
    
    ds_cn = load_dataset('mrhuangtalk/fineweb-edu-chinese-shuffle', split='train', streaming=True)
    ds_en = load_dataset('Josephgflowers/Par-Four-Fineweb-Edu-Fortified', split='train', streaming=True)
    
    iter_cn = iter(ds_cn)
    iter_en = iter(ds_en)
    
    while True:
        try:
            # 依比率隨機或交替取樣
            if torch.rand(1).item() < cn_ratio:
                sample = next(iter_cn)
            else:
                sample = next(iter_en)
            text = sample.get('text', '').strip()
            if text:
                yield text
        except StopIteration:
            break

def create_batch(stream, tokenizer, batch_size=4, max_seq_len=64):
    """從資料串流構建帶有 Padding 的 PyTorch Batch Tensors"""
    batch_std = []
    batch_mask = []
    batch_cj = []
    
    while len(batch_std) < batch_size:
        try:
            text = next(stream)
        except StopIteration:
            break
            
        std_ids, mask, cj_ids = tokenizer.encode(text, max_seq_len=max_seq_len)
        if len(std_ids) >= 4:  # 過濾過短文本
            batch_std.append(std_ids)
            batch_mask.append(mask)
            batch_cj.append(cj_ids)
            
    if not batch_std:
        return None, None, None
        
    # 動態 Padding
    max_len = max(ids.shape[0] for ids in batch_std)
    
    padded_std = []
    padded_mask = []
    padded_cj = []
    
    for i in range(len(batch_std)):
        std = batch_std[i]
        msk = batch_mask[i]
        cj = batch_cj[i]
        
        pad_len = max_len - std.shape[0]
        
        std_pad = torch.cat([std, torch.zeros(pad_len, dtype=torch.long)])
        msk_pad = torch.cat([msk, torch.zeros(pad_len, dtype=torch.bool)])
        cj_pad = torch.cat([cj, torch.full((pad_len, 5), PAD_CJ_ID, dtype=torch.long)])
        
        padded_std.append(std_pad)
        padded_mask.append(msk_pad)
        padded_cj.append(cj_pad)
        
    return (torch.stack(padded_std),
            torch.stack(padded_mask),
            torch.stack(padded_cj))

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用執行裝置: {device}")

    # 1. 初始化 Tokenizer 與模型
    tokenizer = CangjieLlamaTokenizer('pre-train-llama/tokenizer.json')
    model = CangjieTextGenerationModel.from_pretrained('pre-train-llama')
    model.to(device)

    # 2. 階段性參數解凍設定 (Stage 1 vs Stage 2)
    if args.stage == 1:
        print("\n=== [階段 1]: 預熱訓練 (Embedding & 768維 W_1...W_5 投影矩陣) ===")
        print("說明: 凍結 8 層 Decoder 主幹與 LM Head，專注訓練 5 個 768 維 W 矩陣與 27 個倉頡 Token")
        
        for p in model.parameters():
            p.requires_grad = False
            
        # 解凍 W_1...W_5 (5 個 768x768 Linear 矩陣)
        for proj in model.embedding.pos_projections:
            for p in proj.parameters():
                p.requires_grad = True
                
        # 解凍新增的 27 個倉頡 Token 的 Embedding 權重
        model.embedding.embedding.weight.requires_grad = True
        
    else:
        print("\n=== [階段 2]: 全模型端到端聯合微調 (Joint LLM & Tokenizer Training) ===")
        print("說明: 解凍所有 Decoder 層、LM Head、Embedding 與 5 個 768 維 W 矩陣，進行端到端優化")
        for p in model.parameters():
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"可訓練參數總數: {num_trainable:,} / {sum(p.numel() for p in model.parameters()):,}")

    # 3. 初始化數據串流
    stream = get_interleaved_dataset_stream(cn_ratio=args.cn_ratio)

    # 4. 優化器與 CrossEntropy Loss
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 5. 訓練主迴圈
    model.train()
    print(f"\n開始訓練... (共執行 {args.max_steps} 個 Step, Batch Size: {args.batch_size})")
    
    total_loss = 0.0
    for step in range(1, args.max_steps + 1):
        batch_std, batch_mask, batch_cj = create_batch(
            stream, tokenizer, batch_size=args.batch_size, max_seq_len=args.max_seq_len
        )
        
        if batch_std is None:
            print("數據串流結束。")
            break
            
        batch_std = batch_std.to(device)
        batch_mask = batch_mask.to(device)
        batch_cj = batch_cj.to(device)

        # Shift right by 1 for Causal LM target prediction
        inputs_std = batch_std[:, :-1]
        inputs_mask = batch_mask[:, :-1]
        inputs_cj = batch_cj[:, :-1, :]
        targets = batch_std[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs_std, inputs_mask, inputs_cj)

        loss = criterion(logits.reshape(-1, TOTAL_VOCAB_SIZE), targets.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if step % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            print(f"Step {step:5d}/{args.max_steps} | Loss: {avg_loss:.4f}")
            total_loss = 0.0

    # 6. 保存 Checkpoint
    save_filename = f"cangjie_llama_fineweb_stage{args.stage}.pth"
    torch.save(model.state_dict(), save_filename)
    print(f"\n訓練完畢！權重已成功寫入 '{save_filename}'。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FineWeb-Edu 倉頡與 Llama 雙語聯合訓練腳本")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="訓練階段: 1=Embedding預熱, 2=端到端微調")
    parser.add_argument("--max-steps", type=int, default=100, help="訓練 Step 數")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率")
    parser.add_argument("--max-seq-len", type=int, default=128, help="序列長度上限")
    parser.add_argument("--cn-ratio", type=float, default=0.5, help="中文數據混合比例 (0.0~1.0)")
    parser.add_argument("--log-interval", type=int, default=10, help="日誌列印間隔")
    args = parser.parse_args()
    train(args)
