# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from west.dataset.dataset import DataArguments, SpeechDataset
from west.models.model import Model, ModelArgs


@dataclass
class DecodeArguments:
    save_dir: str = field(default=None, metadata={"help": "save dir"})


def main():
    parser = transformers.HfArgumentParser(
        (ModelArgs, DataArguments, DecodeArguments))
    model_args, data_args, decode_args = parser.parse_args_into_dataclasses()
    model = Model.get_model(model_args)
    tokenizer = model.init_tokenizer(model_args)
    test_dataset = SpeechDataset(tokenizer, data_args, inference=True)
    data_loader = DataLoader(test_dataset, collate_fn=lambda x: x[0])
    if torch.cuda.is_available():
        model = model.cuda()
    accelerator = Accelerator()
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    os.makedirs(decode_args.save_dir, exist_ok=True)
    with torch.no_grad():
        for i, item in enumerate(tqdm(data_loader)):
            mel = model.inference(**item)
            print(mel)
            # torch.save(mel[0], os.path.join(decode_args.save_dir, f'{i}.pt'))
            np.save(os.path.join(decode_args.save_dir, f'{i}.npy'),
                    mel[0].detach().cpu().transpose(0, 1).numpy())
            sys.stdout.flush()
            # break


if __name__ == "__main__":
    main()
