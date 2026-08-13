import torch 
import torch.nn.functional as F 
from cangjie_llm import tokenizer, LLM 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = tokenizer()
llm = LLM()
state_dict = torch.load("./src/cangjie_llm/cangjie_epoch_1_latest.pt",map_location = device)
llm.load_state_dict(state_dict)
llm.to(device)
llm.eval()

tok_id_list = [2]

temp = 0.01
max_token = 50 
all_vocab = tok.all_vocab().tolist()


for _ in range(max_token):
    tok_list = torch.tensor([ all_vocab[tok_id] for tok_id in tok_id_list ]).view(1,-1,5).to(device)
    next_tok , _ = llm(tok_list)
    next_tok = next_tok[:,-1,:]
    next_tok = F.softmax(next_tok/temp , dim = -1)
    next_tok_id = torch.multinomial(next_tok , num_samples=1) # gen id 
    tok_id_list.append(next_tok_id)
    print(tok.detokenize(tok_id_list))  # god damn detokenize really need fixing 
