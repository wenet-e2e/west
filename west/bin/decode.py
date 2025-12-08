# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
import json
import os
import sys
from dataclasses import dataclass, field

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, HfArgumentParser

from west.dataset.dataset import DataArguments, SpeechDataset
from west.dataset.extractor import Extractor


@dataclass
class DecodeArguments:
    result_path: str = field(default=None, metadata={"help": "Path to result"})
    model_dir: str = field(default='')


def main():
    parser = HfArgumentParser((DataArguments, DecodeArguments))
    data_args, decode_args = parser.parse_args_into_dataclasses()
    if os.path.isfile(decode_args.model_dir):
        config = AutoConfig.from_pretrained(decode_args.model_dir)
        model = AutoModel.from_config(config)
    else:
        model = AutoModel.from_pretrained(decode_args.model_dir)
    tokenizer = model.init_tokenizer()
    extractor = Extractor.get_class(model.model_type)(tokenizer,
                                                      model.config,
                                                      inference=True)
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
            generated_ids = model.generate(**item)
            text = tokenizer.batch_decode(generated_ids,
                                          skip_special_tokens=True)
            print(text)
            for t in text:
                t = t.replace('\n', ' ')
                item = {'txt': t}
                fid.write(json.dumps(item, ensure_ascii=False) + '\n')
            sys.stdout.flush()
    fid.close()


if __name__ == "__main__":
    main()
