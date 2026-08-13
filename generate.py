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

tok_list = [[[tok.vocab["[BOS]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"], tok.vocab["[PAD]"]]]]
tok_list = torch.tensor(tok_list).to(device)
temp = 0.01
max_token = 20 
all_vocab = tok.all_vocab()


for _ in range(max_token):
    next_tok , _ = llm(tok_list)
    next_tok = next_tok[:,-1,:]
    next_tok = F.softmax(next_tok/temp , dim = -1)
    next_tok = torch.multinomial(next_tok , num_samples=1)
    next_tok = all_vocab[next_tok.view(-1),:]
    tok_list = torch.cat([token_list , next_tok] , dim = 1).to(device)
    print(tok.detokenize(tok_list))
