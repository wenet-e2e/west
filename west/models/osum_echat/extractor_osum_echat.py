# Copyright (c) 2025 Xuelong Geng(xlgeng@mail.nwpu.edu.cn)

import math
import random

import torch
import wenet
from gxl_ai_utils.utils import utils_file
from transformers.trainer_pt_utils import LabelSmoother

from west.dataset.extractor import Extractor


class ExtractorOSUM(Extractor):
    model_type = "osum"
    fields_batch_static = {'audio_offsets'}
    fields_batch_dynamic = {'audio_features', 'input_ids', 'labels'}
    fields_pack_offset = {'audio_offsets'}

    def __init__(self, tokenizer, model_config, inference=False):
        super().__init__(tokenizer, model_config, inference)
        self.compute_feature, self.feature_dim = wenet.load_feature(
            self.model_config.wenet_model_name_or_path
        )
        self.ds_rate = (self.model_config.encoder_ds_rate *
                        self.model_config.encoder_projector_ds_rate)

        self.global_prompt_dict = utils_file.load_dict_from_yaml(
            model_config.prompt_conf_path)

    def extract(self, item):
        """
        {'wav': wav_path, 'txt': text, 'task': '<TRANSCRIBE>'}
        """
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        t0 = ''
        if 'task' in item:
            task_name = item['task']
            try:
                random_index = random.randint(
                    0, len(self.global_prompt_dict[task_name]) - 1)
                prompt = self.global_prompt_dict[task_name][random_index]
                if prompt != "<no_prompt>":
                    t0 = prompt
            except Exception as e:
                print(f"Error: {e}")
        else:
            task_name = '<TRANSCRIBE>'  # default: speech recognition
            try:
                random_index = random.randint(
                    0, len(self.global_prompt_dict[task_name]) - 1)
                prompt = self.global_prompt_dict[task_name][random_index]
                t0 = prompt
            except Exception as e:
                print(f"Error: {e}")

        mel = self.compute_feature(item['wav'])
        ids_audio = [0] * (math.ceil(mel.size(0) / self.ds_rate) + 2)

        ids0 = self.tokenizer.encode(t0)
        ids = ids0 + ids_audio
        tgt = [IGNORE_TOKEN_ID] * len(ids)

        if not self.inference:
            ids1 = self.tokenizer.encode(item['txt'] + "<|endoftext|>")
            ids = ids + ids1
            tgt = tgt + ids1

        input_ids = torch.tensor(ids, dtype=torch.int)
        tgt_ids = torch.tensor(tgt, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': tgt_ids,
            'audio_features': mel,
            'audio_offsets': len(ids0),
        }


class ExtractorOSUMEChat(ExtractorOSUM):
    model_type = "osum_echat"

    def __init__(self, tokenizer, model_config, inference=False):
        super().__init__(tokenizer, model_config, inference)
        # TODO(Xuelong Geng): Complete the design of extractor

    def extract(self, item):
        """
        TODO(Xuelong Geng): Complete the function
        """
