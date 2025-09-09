# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import io
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.distributed as dist
import webdataset as wds
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers.trainer_pt_utils import LabelSmoother

from west.dataset.extractor import Extractor


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    batch_size: int = field(default=1, metadata={"help": "batch size"})
    pack_size: int = field(
        default=0,
        metadata={
            "help":
            "size for sequence pack, it will override any value"
            "given in batch_size"
        })
    num_data_cycles: int = field(
        default=1,
        metadata={
            "help":
            "repeating read `num_data_cycles` times to avoid uneven data in "
            "training, especically for tar(shard) training. Typically you can  "
            "set it to the training epochs"
        })


class SpeechDataset(IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        extractor: Extractor,
        data_args: DataArguments,
        inference: bool = False,
    ):
        super(SpeechDataset, self).__init__()
        self.data_path = data_args.data_path
        self.extractor = extractor
        self.tokenizer = extractor.tokenizer
        self.inference = extractor.inference
        if data_args.pack_size > 0:
            self.mode = 'pack'
            self.pack_size = data_args.pack_size
        else:
            self.mode = 'static'
            self.batch_size = data_args.batch_size

        self.data_args = data_args
        self.data_lists = []
        with open(self.data_path, "r") as f:
            for i, line in enumerate(f):
                self.data_lists.append(line.strip())

    def set_epoch(self, epoch):
        if not self.inference:
            # Set epoch as random seed, which ensures we have the same shuffle
            # list in training for different rank & workers
            local_random = random.Random(epoch)
            local_random.shuffle(self.data_lists)

    def _read_one(self):
        try:
            # world_size = dist.get_world_size()
            rank = dist.get_rank()
        except Exception:
            # world_size = 1
            rank = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        lists = self.data_lists
        # ATTENTION !!!! huggingface accelerator shards the data autotmaticly,
        # otherwise, we should split the data by the `world_size` and `rank` as
        # lists = self.data_lists[rank::world_size]
        # please see:
        # https://github.com/huggingface/accelerate/blob/v1.8.1/src/accelerate/data_loader.py#L1210  # noqa
        # Devide by reading workers
        lists = lists[worker_id::num_workers]
        raw = self.data_path.endswith('.jsonl')
        for i in range(self.data_args.num_data_cycles):
            logging.info(f'Data start iter epoch {i} rank {rank} '
                         f'worker {worker_id} data_size {len(lists)}')
            for line in lists:
                if raw:  # raw json data
                    yield json.loads(line)
                else:  # shard(tar) list data
                    src = [{'url': line}]
                    try:
                        data = wds.tarfile_samples(src)
                        for x in data:
                            try:
                                x['txt'] = x['txt'].decode('utf8')
                                x['wav'] = io.BytesIO(x['wav'])
                                yield x
                            except Exception:
                                logging.info(f'Dataset decode error, {line}')
                                continue
                    except Exception:
                        logging.info(f'Dataset parsing error, {line}')
                        continue

    def _pack_sequence(self, seqs):
        """
        Our base LLM will apply `shift_labels` on the labels. Assume we have:
        input_ids: <sos> a  b   c      <eos>  <sos> x y  z      <eos>
                   a     b  c   <eos>  N      x     y z  <eos>  N
                   N     a  b   c      <eos>  N     x y  z      <eos>
        The target should like above after `shift_labels`, where N is for
        ignore_index, we should ignore the target when the input_ids is <eos>

        """
        pack_size = self.pack_size
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        ret = {}
        ret['input_ids'] = torch.tensor([0] * pack_size, dtype=torch.int)
        ret['labels'] = torch.tensor([IGNORE_TOKEN_ID] * pack_size,
                                     dtype=torch.long)
        ret['position_ids'] = torch.tensor([0] * pack_size, dtype=torch.int)
        for k in self.extractor.fields_pack_offset:
            ret[k] = torch.tensor([0] * len(seqs), dtype=torch.int)
        offset = 0
        for i, seq in enumerate(seqs):
            for k in self.extractor.fields_pack_offset:
                ret[k][i] = offset + seq[k]
            seq_len = len(seq['input_ids'])
            ret['input_ids'][offset:offset + seq_len] = seq['input_ids']
            ret['labels'][offset] = IGNORE_TOKEN_ID
            ret['labels'][offset + 1:offset + seq_len] = seq['labels'][1:]
            ret['position_ids'][offset:offset + seq_len] = torch.arange(
                seq_len, dtype=torch.int)
            offset += seq_len
        ret['batch_idx'] = torch.tensor([0] * len(seqs), dtype=torch.int)
        ret['input_ids'] = ret['input_ids'].unsqueeze(0)
        ret['labels'] = ret['labels'].unsqueeze(0)
        ret['position_ids'] = ret['position_ids'].unsqueeze(0)

        ret = ret | self._batch(seqs, pack=True)
        return ret

    def _batch(self, seqs, pack=False):
        """ If pack is true, exclude the files for pack
        """
        ret = {}
        if not pack:
            fields_dynamic = self.extractor.fields_batch_dynamic
            fields_static = self.extractor.fields_batch_static
        else:
            fields_dynamic = self.extractor.fields_batch_dynamic - {
                'input_ids', 'labels'
            }
            fields_static = self.extractor.fields_batch_static - \
                self.extractor.fields_pack_offset
        for k in fields_dynamic:
            if k == 'input_ids':
                padding_value = self.tokenizer.pad_token_id
            elif k == 'labels':
                padding_value = LabelSmoother.ignore_index
            else:
                padding_value = 0
            ret[k] = pad_sequence([s[k] for s in seqs],
                                  batch_first=True,
                                  padding_value=padding_value)
            if k not in ['input_ids', 'labels']:
                ret[k + '_lengths'] = torch.tensor([s[k].size(0) for s in seqs],
                                                   dtype=torch.int)
            if k == 'input_ids':
                ret['attention_mask'] = ret['input_ids'].ne(
                    self.tokenizer.pad_token_id)
        for k in fields_static:
            ret[k] = torch.tensor([s[k] for s in seqs], dtype=torch.int)
        if not pack:
            ret['batch_idx'] = torch.tensor(list(range(len(seqs))),
                                            dtype=torch.int)
        return ret

    def __iter__(self) -> Dict[str, torch.Tensor]:
        buffer = []
        total_length = 0
        for item in self._read_one():
            data = self.extractor.extract(item)
            # TODO(Binbin Zhang): move filter to extractor, if the data is not
            # valid, extractor should return None
            if data is None:
                continue
            if self.mode == 'static' and len(buffer) == self.batch_size:
                yield self._batch(buffer)
                buffer = []
                total_length = 0
            elif self.mode == 'pack' and total_length + len(
                    data['input_ids']) >= self.pack_size:
                yield self._pack_sequence(buffer)
                buffer = []
                total_length = 0
            buffer.append(data)
            if 'input_ids' in data:
                total_length += len(data['input_ids'])
        if self.mode == 'static':
            yield self._batch(buffer)
        else:
            yield self._pack_sequence(buffer)


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/jfs-hdfs/user/binbin.zhang/huggingface/hub/Qwen2-1.5B-Instruct')
    tokenizer.bos_token = tokenizer.eos_token
    print(tokenizer.bos_token_id)
    data_args = DataArguments
    data_args.data_path = 'data/train.jsonl'
    data_args.extractor_type = 'tts_codec'
    dataset = SpeechDataset(tokenizer, data_args)
    for i, x in enumerate(dataset):
        print(x)
        if i > 0:
            break
