# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
import sys
from dataclasses import dataclass, field

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, HfArgumentParser

from west.dataset.dataset import DataArguments, SpeechDataset
from west.dataset.extractor import Extractor


@dataclass
class DecodeArguments:
    llm_type: str = 'qwen2'
    max_new_tokens: int = 50
    num_beams: int = 1
    result_path: str = field(default=None, metadata={"help": "Path to result"})
    model_dir: str = field(default='')


def main():
    parser = HfArgumentParser((DataArguments, DecodeArguments))
    data_args, decode_args = parser.parse_args_into_dataclasses()
    model = AutoModel.from_pretrained(decode_args.model_dir)
    tokenizer = model.init_tokenizer()
    extractor = Extractor.get_class(model.model_type)(tokenizer, inference=True)
    if decode_args.llm_type == 'qwen2':
        eos_token_id = tokenizer.convert_tokens_to_ids(
            ['<|endoftext|>', '<|im_end|>'])
    else:
        eos_token_id = tokenizer.convert_tokens_to_ids(
            ['<|end_of_text|>', '<|eot_id|>'])
    print('eos_token_id', eos_token_id)
    test_dataset = SpeechDataset(extractor, data_args)
    data_loader = DataLoader(test_dataset, collate_fn=lambda x: x[0])
    if torch.cuda.is_available():
        model = model.cuda()
    accelerator = Accelerator()
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    fid = open(decode_args.result_path, 'w', encoding='utf8')
    with torch.no_grad():
        for i, item in enumerate(tqdm(data_loader)):
            generated_ids = model.generate(**item,
                                           eos_token_id=eos_token_id,
                                           decode_config=decode_args)
            text = tokenizer.batch_decode(generated_ids,
                                          skip_special_tokens=True)
            print(text)
            for t in text:
                t = t.replace('\n', ' ')
                fid.write(t + '\n')
            sys.stdout.flush()
    fid.close()


if __name__ == "__main__":
    main()
