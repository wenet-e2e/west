# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import io
import json
import random
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.distributed as dist
import transformers
import webdataset as wds
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers.trainer_pt_utils import LabelSmoother

from west.dataset.extractor import ExtractorFactory


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
    max_speech_frames: int = 1000
    extractor_type: str = field(
        default="asr_wenet",
        metadata={"help": "extractor type, 'asr_wenet' or 'tts_codec'"})


class SpeechDataset(IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        inference: bool = False,
    ):
        super(SpeechDataset, self).__init__()
        self.data_path = data_args.data_path
        self.tokenizer = tokenizer
        self.inference = inference
        if data_args.pack_size > 0:
            self.mode = 'pack'
            self.pack_size = data_args.pack_size
        else:
            self.mode = 'static'
            self.batch_size = data_args.batch_size
        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        except Exception:
            self.world_size = 1
            self.rank = 0
        self.data_args = data_args
        self.extractor = ExtractorFactory.create(data_args.extractor_type)(
            tokenizer=tokenizer,
            inference=inference,
        )
        self.data_lists = []
        with open(self.data_path, "r") as f:
            for i, line in enumerate(f):
                if i % self.world_size == self.rank:
                    self.data_lists.append(line.strip())
        if not self.inference:
            random.shuffle(self.data_lists)

    def _read_one(self):
        raw = self.data_path.endswith('.jsonl')
        for i, line in enumerate(self.data_lists):
            if raw:  # raw json data
                yield json.loads(line)
            else:  # shard(tar) list data
                src = [{'url': line}]
                data = wds.tarfile_samples(src)
                for x in data:
                    x['txt'] = x['txt'].decode('utf8')
                    x['wav'] = io.BytesIO(x['wav'])
                    yield x

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
        input_ids = torch.tensor([0] * pack_size, dtype=torch.int)
        labels = torch.tensor([IGNORE_TOKEN_ID] * pack_size, dtype=torch.long)
        position_ids = torch.tensor([0] * pack_size, dtype=torch.int)
        batch_idx = torch.tensor([0] * len(seqs), dtype=torch.int)
        audio_offsets = torch.tensor([0] * len(seqs), dtype=torch.int)
        seq_ids = torch.tensor([0] * pack_size, dtype=torch.int)
        audio_features = []
        offset = 0
        cu_seq_lens = [0]
        max_length = 0
        for i, seq in enumerate(seqs):
            audio_offsets[i] = offset + seq['offset']
            seq_len = len(seq['input_ids'])
            input_ids[offset:offset + seq_len] = seq['input_ids']
            labels[offset] = IGNORE_TOKEN_ID
            labels[offset + 1:offset + seq_len] = seq['labels'][1:]
            cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
            max_length = max(max_length, seq_len)
            position_ids[offset:offset + seq_len] = torch.arange(
                seq_len, dtype=torch.int)
            seq_ids[offset:offset + seq_len] = i + 1
            audio_features.append(seq['mel'])
            offset += seq_len
        # labels[offset] = IGNORE_TOKEN_ID
        audio_feature_lengths = torch.tensor(
            [t.size(0) for t in audio_features], dtype=torch.int)
        audio_features = pad_sequence(audio_features, batch_first=True)
        cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int)
        return {
            'input_ids': input_ids.unsqueeze(0),
            'labels': labels.unsqueeze(0),
            # 'attention_mask': seq_ids,  # only used for flex attention
            'position_ids': position_ids.unsqueeze(0),
            'audio_offsets': audio_offsets,
            'audio_features': audio_features,
            'audio_feature_lengths': audio_feature_lengths,
            'batch_idx': batch_idx,
        }

    def _batch(self, seqs):
        audio_features = [s['mel'] for s in seqs]
        audio_feature_lengths = torch.tensor(
            [t.size(0) for t in audio_features], dtype=torch.int)
        audio_features = pad_sequence(audio_features, batch_first=True)
        input_ids = pad_sequence([s['input_ids'] for s in seqs],
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence([s['labels'] for s in seqs],
                              batch_first=True,
                              padding_value=LabelSmoother.ignore_index)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        audio_offsets = torch.tensor([s['offset'] for s in seqs],
                                     dtype=torch.int)
        batch_idx = torch.tensor(list(range(len(seqs))), dtype=torch.int)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'audio_offsets': audio_offsets,
            'audio_features': audio_features,
            'audio_feature_lengths': audio_feature_lengths,
            'batch_idx': batch_idx,
        }

    def __iter__(self) -> Dict[str, torch.Tensor]:
        buffer = []
        total_length = 0
        for item in self._read_one():
            data = self.extractor.extract(item)
            if data['mel'].size(0) > self.data_args.max_speech_frames and \
               not self.inference:
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
