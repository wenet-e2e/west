import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

flash_attn = True

model_dir = 'Qwen/Qwen2-1.5B-Instruct'
flash_kwargs = {
    'attn_implementation': 'flash_attention_2'
} if flash_attn else {}
model = AutoModelForCausalLM.from_pretrained(model_dir,
                                             device_map="auto",
                                             torch_dtype='auto',
                                             **flash_kwargs)

tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                          use_fast=False,
                                          padding_side="right")

token_len = 4
ids = list(range(0, token_len))
# ids = [0, 1, 2, tokenizer.eos_token_id]
input_ids0 = torch.tensor([ids], dtype=torch.long)
input_ids1 = torch.tensor([ids, ids], dtype=torch.long)
out0 = model(input_ids0.cuda(), labels=input_ids0.cuda())
out1 = model(input_ids1.cuda(), labels=input_ids1.cuda())

input_ids2 = torch.tensor([ids + ids + [1000, 1000]], dtype=torch.long)
position_ids = torch.tensor([ids + ids + [0, 0]], dtype=torch.int)
# cu_seq_lens = torch.tensor([0, 4, 8], dtype=torch.int).unsqueeze(0)
# attention_mask2 = torch.zeros((8, 8), dtype=torch.bool)
# for i in range(4):
#     attention_mask2[i, 0:i + 1] = True
# for i in range(4):
#     attention_mask2[i + 4, 4:4 + i + 1] = True
# print(attention_mask2)
# attention_mask2 = attention_mask2.unsqueeze(0)
if flash_attn:
    kwargs = {}
# else:
#     kwargs = {'attention_mask': attention_mask2.cuda()}

out2 = model(input_ids2.cuda(),
             labels=input_ids2.cuda(),
             position_ids=position_ids.cuda(),
             **kwargs)
sim0 = F.cosine_similarity(out1.logits[0, :, :], out1.logits[1, :, :])
print(sim0)
sim1 = F.cosine_similarity(out1.logits[0, :, :], out2.logits[0, :token_len, :])
print(sim1)
sim2 = F.cosine_similarity(out1.logits[0, :, :], out2.logits[0,
                                                             token_len:-2, :])
print(sim2)

print(out0.loss)
print(out1.loss)
print(out2.loss)
