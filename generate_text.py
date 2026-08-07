import torch
import torch.nn.functional as F
from cangjie_tokenizer import CangjieLlamaTokenizer
from modeling_cangjie_llama import CangjieTextGenerationModel, TOTAL_VOCAB_SIZE

def generate(model, tokenizer, prompt, max_new_tokens=40, temperature=0.7, device='cuda'):
    std_ids, mask, cj_ids = tokenizer.encode(prompt)
    std_ids = std_ids.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    cj_ids = cj_ids.unsqueeze(0).to(device)
    
    generated_tokens = std_ids[0].tolist()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(std_ids, mask, cj_ids)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            
        generated_tokens.append(next_id)
        if next_id == 1:  # [eos]
            break
            
        # Append next step
        next_std = torch.tensor([[next_id]], device=device)
        next_mask = torch.tensor([[False]], device=device)
        next_cj = torch.full((1, 1, 5), 10026, device=device, dtype=torch.long)
        
        std_ids = torch.cat([std_ids, next_std], dim=1)
        mask = torch.cat([mask, next_mask], dim=1)
        cj_ids = torch.cat([cj_ids, next_cj], dim=1)

    # Decode standard token sequence
    try:
        decoded_text = tokenizer.base_tokenizer.decode(generated_tokens)
    except Exception:
        decoded_text = str(generated_tokens)
    return decoded_text

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CangjieLlamaTokenizer('pre-train-llama/tokenizer.json')
    model = CangjieTextGenerationModel.from_pretrained('pre-train-llama')
    
    # Load Stage 2 checkpoint
    checkpoint_path = 'cangjie_llama_fineweb_stage2.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    prompts = [
        "Hello World,",
        "Once upon a time,",
        "高校人才培養",
        "日月星辰"
    ]
    
    print("\n=== 模型生成測試 ===")
    for p in prompts:
        gen = generate(model, tokenizer, p, max_new_tokens=30, temperature=0.7, device=device)
        print(f"\nPrompt: {p}")
        print(f"Output: {gen}")
