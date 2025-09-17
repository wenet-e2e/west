# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1]
num_speech_tokens = int(sys.argv[2])
dst_dir = sys.argv[3]

model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

special_audio_tokens = ['<|audio|>', '<|audio_bos|>', '<|audio_eos|>']

special_tokens_dict = {'additional_special_tokens': special_audio_tokens}

num_added = tokenizer.add_special_tokens(special_tokens_dict)
print(f'Adding {num_added} special audio tokens')

# Make sure the last `num_speech_tokens` are speech tokens
audio_tokens = [f'<|speech_{i}|>' for i in range(num_speech_tokens)]
num_added = tokenizer.add_tokens(audio_tokens)
print(f'Adding {num_added} audio tokens')

model.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained(dst_dir)
model.save_pretrained(dst_dir)
