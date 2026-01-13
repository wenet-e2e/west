# -*- coding: utf-8 -*-
"""
@Author : songyd, chenhj
@File   : modeling_goat_slm.py
"""
import time
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from transformers import CONFIG_MAPPING, PreTrainedModel, WhisperConfig
from transformers.generation.streamers import BaseStreamer

from west.utils.mask import lengths_to_padding_mask

from .configuration_goat_slm import GOATSLMConfig
from .configuration_transformer_adapter import AdapterConfig
from .modeling_qwen2 import Qwen2ForCausalLM
from .modeling_qwen3 import Qwen3ForCausalLM
from .modeling_telechat3 import Telechat3Config, Telechat3ForCausalLM
from .modeling_transformer_adapter import AdapterModel
from .modeling_whisper_encoder import WhisperEncoder

IGNORE_INDEX = -100


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
            self,
            in_channels: int,
            mid_channels: int,
            out_channels: int,
            kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            ) for i, k in enumerate(kernel_sizes))

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


class GOATSLMModel(PreTrainedModel):
    config_class = GOATSLMConfig
    base_model_prefix = "goat_slm"

    def __init__(self, config):
        super().__init__(config)
        self.whisper_config = WhisperConfig(**config.whisper_config)
        model_type = config.llm_config.get("model_type")
        # print(f"model_type: {model_type}")
        if model_type in CONFIG_MAPPING:  # qwen3 or qwen2
            config_class = CONFIG_MAPPING[model_type]
            llm_config = config_class.from_dict(config.llm_config)
        elif model_type == "telechat3":
            llm_config = Telechat3Config.from_dict(config.llm_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.llm_config = llm_config

        # force to use flash attention2
        self.llm_config._attn_implementation = "flash_attention_2"
        self.llm_config.torch_dtype = torch.bfloat16

        self.whisper_model = WhisperEncoder(self.whisper_config)
        if model_type == 'qwen2':
            self.llm_model = Qwen2ForCausalLM(self.llm_config)
            self.llm_model.speech_generator = Qwen2ForCausalLM(self.llm_config)
        elif model_type == 'qwen3':
            self.llm_model = Qwen3ForCausalLM(self.llm_config)
            self.llm_model.speech_generator = Qwen3ForCausalLM(self.llm_config)
        elif model_type == "telechat3":
            self.llm_model = Telechat3ForCausalLM(self.llm_config)
            self.llm_model.speech_generator = Telechat3ForCausalLM(
                self.llm_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # adapter stuff
        self.adapter_config = AdapterConfig(**config.adapter_config)
        self.adapter_config.torch_dtype = torch.bfloat16
        self.adapter = AdapterModel(self.adapter_config)

        in_d = self.whisper_config.d_model
        out_d = self.adapter_config.hidden_size
        self.subsampler = Conv1dSubsampler(
            in_d,
            2 * in_d,
            out_d,
            [int(k) for k in config.conv_kernel_sizes.split(",")],
        )
        self.speech_ln = torch.nn.LayerNorm(out_d, 1e-5, True)

        # init speech generator
        self.initialize_speech_generator(self.llm_config)
        self.num_tok_per_group = 4
        self.llm_model.add_module(
            'emb_unit',
            nn.Embedding(self.llm_config.unit_vocab_size + 1,
                         self.llm_config.hidden_size))
        self.llm_model.add_module(
            'head_unit',
            nn.ModuleList([
                nn.Linear(self.llm_config.hidden_size // 2 if model_type
                          == "telechat3" else self.llm_config.hidden_size,
                          self.llm_config.unit_vocab_size + 1,
                          bias=False) for _ in range(self.num_tok_per_group)
            ]))

        self.llm_model.freeze_layer = self.llm_config.freeze_layer

    def initialize_speech_generator(self, model_args):
        self.llm_config.unit_vocab_size = getattr(model_args, 'unit_vocab_size',
                                                  4097)
        self.llm_config.num_tok_per_group = getattr(model_args,
                                                    'num_tok_per_group', 4)

    def get_speech_features(self, speech_values, speech_attention_mask):
        w2v_args = {
            "input_features": speech_values,
            "attention_mask": speech_attention_mask,
        }
        output = self.whisper_model(**w2v_args)
        speech_embeds = output.last_hidden_state  # B x T x C
        speech_lengths = output.output_lengths

        speech_embeds, speech_lengths = self.subsampler(speech_embeds,
                                                        speech_lengths)
        speech_embeds = speech_embeds.transpose(0, 1)  # T x B x C -> B x T x C
        speech_padding_mask = lengths_to_padding_mask(speech_lengths)
        speech_atts = ~speech_padding_mask

        speech_embeds = self.adapter(inputs_embeds=speech_embeds,
                                     attention_mask=speech_atts,
                                     return_dict=True).last_hidden_state
        speech_embeds = self.speech_ln(speech_embeds)

        return speech_embeds, speech_atts

    def prepare_inputs_labels_for_speech_and_text(self,
                                                  input_ids,
                                                  query_ids,
                                                  suffix_input_ids,
                                                  speech_values,
                                                  speech_attention_mask,
                                                  speech_llm_input_ids=None):
        inputs_embeds, attention_mask = [], []

        prefix_embeds = self.llm_model.get_input_embeddings()(input_ids)
        prefix_attns = torch.ones(prefix_embeds.size(0),
                                  prefix_embeds.size(1),
                                  dtype=torch.long).to(prefix_embeds.device)

        suffix_embeds = self.llm_model.get_input_embeddings()(suffix_input_ids)
        suffix_attns = torch.ones(suffix_embeds.size(0),
                                  suffix_embeds.size(1),
                                  dtype=torch.long).to(suffix_embeds.device)

        inputs_embeds.append(prefix_embeds)
        attention_mask.append(prefix_attns)

        query_embeds = self.llm_model.get_input_embeddings()(query_ids)
        query_attns = torch.ones(query_embeds.size(0),
                                 query_embeds.size(1),
                                 dtype=torch.long).to(query_embeds.device)
        inputs_embeds.append(query_embeds)
        attention_mask.append(query_attns)

        if speech_values is not None:
            if isinstance(speech_values, np.ndarray):
                input_names = ['feature', 'feature_lens']
                ort_inputs = {}
                input_numpy = [
                    speech_values,
                    np.array([np.sum(speech_attention_mask[0])], dtype=np.int64)
                ]
                for idx, name in enumerate(input_names):
                    ort_inputs[name] = input_numpy[idx]
                start_time = time.time()
                ort_outs = self.onnx_session.run(None, ort_inputs)
                end_time = time.time()
                print("onnx runtime:", end_time - start_time)
                speech_embeds, speech_attention_mask = ort_outs
                speech_embeds = torch.from_numpy(speech_embeds).to(
                    device=prefix_embeds.device, dtype=prefix_embeds.dtype)
                speech_attention_mask = torch.from_numpy(
                    speech_attention_mask).to(device=prefix_embeds.device,
                                              dtype=prefix_embeds.dtype)
            else:
                speech_embeds, speech_attention_mask = self.get_speech_features(
                    speech_values, speech_attention_mask)

            inputs_embeds.append(speech_embeds)
            attention_mask.append(speech_attention_mask)

        inputs_embeds.append(suffix_embeds)
        attention_mask.append(suffix_attns)

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)

        if speech_llm_input_ids is not None:
            speech_llm_prefix_embeds = self.llm_model.get_input_embeddings()(
                speech_llm_input_ids)
            speech_llm_prefix_attention_mask = torch.ones(
                speech_llm_prefix_embeds.size(0),
                speech_llm_prefix_embeds.size(1),
                dtype=torch.long).to(speech_llm_prefix_embeds.device)
            text_llm_prefix_len = prefix_embeds.size(1)
        else:
            speech_llm_prefix_attention_mask = None
            speech_llm_prefix_embeds = None
            text_llm_prefix_len = None

        return (
            inputs_embeds,
            attention_mask,
            speech_llm_prefix_embeds,
            speech_llm_prefix_attention_mask,
            text_llm_prefix_len,
        )

    @torch.no_grad()
    def generate_custom(self,
                        input_ids,
                        suffix_input_ids,
                        query_ids=None,
                        speech_values=None,
                        speech_attention_mask=None,
                        generation_config=None,
                        streamer: Optional["BaseStreamer"] = None,
                        text_label=None,
                        is_teacher_forcing=False,
                        prompt_unit=None,
                        speech_llm_input_ids=None,
                        max_speech_unit_length=50 * 35,
                        is_speech_generate_run=True,
                        enable_thinking=False):
        assert (not is_teacher_forcing) or (
            is_teacher_forcing and text_label is not None
        ), 'teacher forcing is True, but text label is None!!'

        inputs_embeds, attention_mask, speech_llm_prefix_embeds, speech_llm_prefix_attention_mask, text_llm_prefix_len = self.prepare_inputs_labels_for_speech_and_text(  # noqa
            input_ids, query_ids, suffix_input_ids, speech_values,
            speech_attention_mask, speech_llm_input_ids)

        custom_params = {
            "speech_llm_prefix_embeds": speech_llm_prefix_embeds,
            "speech_llm_prefix_attention_mask":
            speech_llm_prefix_attention_mask,
            "text_llm_prefix_len": text_llm_prefix_len,
            "text_label": text_label,
            "is_teacher_forcing": is_teacher_forcing,
            "prompt_unit": prompt_unit,
            'max_speech_unit_length': max_speech_unit_length,
            'is_speech_generate_run': is_speech_generate_run,
            'enable_thinking': enable_thinking
        }

        outputs = self.llm_model.generate(inputs_embeds=inputs_embeds,
                                          streamer=streamer,
                                          generation_config=generation_config,
                                          attention_mask=attention_mask,
                                          custom_params=custom_params)

        return outputs.sequences, outputs.speech_units

    @torch.no_grad()
    def generate_custom_for_multi_turn(self,
                                       history,
                                       tokenizer,
                                       generation_config=None,
                                       max_speech_unit_length=30 * 50,
                                       is_speech_generate_run=True,
                                       enable_thinking=False,
                                       device='cuda'):
        inputs_embeds = []

        # _multi_turn_indices = []
        index = 0
        text_llm_prefix_len = 0
        speech_llm_prefix_embeds = speech_llm_prefix_attention_mask = None
        print('********************* history start ***************************')
        for item in history:
            if item['role'] == 'text_system':
                for content_item in item['content']:
                    if content_item['type'] == 'text':
                        input_ids = tokenizer.encode(content_item['text'])
                        text_llm_prefix_len = len(input_ids)
                        input_ids = torch.tensor(input_ids,
                                                 dtype=torch.int,
                                                 device=device).unsqueeze(0)
                        embeds = self.llm_model.get_input_embeddings()(
                            input_ids)
                        inputs_embeds.append(embeds)
                        print(f"{content_item['text']}")
            elif item['role'] == 'speech_system':
                for content_item in item['content']:
                    if content_item['type'] == 'text':
                        input_ids = tokenizer.encode(content_item['text'])
                        input_ids = torch.tensor(input_ids,
                                                 dtype=torch.int,
                                                 device=device).unsqueeze(0)
                        speech_llm_prefix_embeds = self.llm_model.get_input_embeddings()(input_ids)
                        speech_llm_prefix_attention_mask = torch.ones(
                            speech_llm_prefix_embeds.size(0),
                            speech_llm_prefix_embeds.size(1),
                            dtype=torch.long).to(speech_llm_prefix_embeds.device)
            elif item['role'] == 'user':
                index = 0
                for content_item in item['content']:
                    if content_item['type'] == 'text':
                        input_ids = tokenizer.encode(content_item['text'])
                        input_ids = torch.tensor(input_ids,
                                                 dtype=torch.int,
                                                 device=device).unsqueeze(0)
                        embeds = self.llm_model.get_input_embeddings()(input_ids)
                        inputs_embeds.append(embeds)
                        index += input_ids.size(-1)
                        print(f"{content_item['text']}")
                    elif content_item['type'] == 'audio':
                        speech_values, speech_attention_mask = content_item['audio']
                        speech_embeds, _ = self.get_speech_features(
                            speech_values, speech_attention_mask)
                        inputs_embeds.append(speech_embeds)
                        index += speech_embeds.size(1)
                        print(f"{speech_values.size()}")
            elif item['role'] == 'assistant':
                for content_item in item['content']:
                    if content_item['type'] == 'text':
                        input_ids = tokenizer.encode(content_item['text'])
                        input_ids = torch.tensor(input_ids,
                                                 dtype=torch.int,
                                                 device=device).unsqueeze(0)
                        embeds = self.llm_model.get_input_embeddings()(
                            input_ids)
                        inputs_embeds.append(embeds)
                        index += input_ids.size(-1)
                        print(f"{content_item['text']}")
                if not item['thinking']:
                    input_ids = tokenizer.encode('<think>\n\n</think>\n\n')
                    input_ids = torch.tensor(input_ids,
                                             dtype=torch.int,
                                             device=device).unsqueeze(0)
                    embeds = self.llm_model.get_input_embeddings()(input_ids)
                    inputs_embeds.append(embeds)
                    index += input_ids.size(-1)
                    print("<think>\n\n</think>\n\n")
        print('********************* history end ***************************')
        multi_turn_for_decoder_indices = [[
            0, text_llm_prefix_len if speech_llm_prefix_embeds is None else
            speech_llm_prefix_embeds.size(1)
        ], [-index, -1]]
        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        print(f"inputs_embeds: {inputs_embeds.shape}")
        # print(f"multi_turn_for_decoder_indices: {multi_turn_for_decoder_indices}")
        print(f"text_llm_prefix_len: {text_llm_prefix_len}")
        print(
            f"speech_llm_prefix_len: {speech_llm_prefix_embeds.size(1) if speech_llm_prefix_embeds is not None else 0}"
        )

        custom_params = {
            "speech_llm_prefix_embeds": speech_llm_prefix_embeds,
            "speech_llm_prefix_attention_mask":
            speech_llm_prefix_attention_mask,
            "text_llm_prefix_len": text_llm_prefix_len,
            "text_label": None,
            "is_teacher_forcing": False,
            "prompt_unit": None,
            'max_speech_unit_length': max_speech_unit_length,
            'is_speech_generate_run': is_speech_generate_run,
            'multi_turn_for_decoder_indices': multi_turn_for_decoder_indices,
            'enable_thinking': enable_thinking
        }

        outputs = self.llm_model.generate(inputs_embeds=inputs_embeds,
                                          generation_config=generation_config,
                                          custom_params=custom_params)

        return outputs.sequences, outputs.speech_units

    @torch.no_grad()
    def generate(self,
                 input_ids,
                 suffix_input_ids,
                 speech_values=None,
                 speech_attention_mask=None,
                 generation_config=None):
        inputs_embeds, attention_mask = [], []

        prefix_embeds = self.llm_model.get_input_embeddings()(input_ids)
        prefix_attns = torch.ones(prefix_embeds.size(0),
                                  prefix_embeds.size(1),
                                  dtype=torch.long).to(prefix_embeds.device)
        inputs_embeds.append(prefix_embeds)
        attention_mask.append(prefix_attns)

        if speech_values is not None:
            speech_embeds, speech_attention_mask = self.get_speech_features(
                speech_values, speech_attention_mask)
            inputs_embeds.append(speech_embeds)
            attention_mask.append(speech_attention_mask)

        suffix_embeds = self.llm_model.get_input_embeddings()(suffix_input_ids)
        suffix_attns = torch.ones(suffix_embeds.size(0),
                                  suffix_embeds.size(1),
                                  dtype=torch.long).to(suffix_embeds.device)
        inputs_embeds.append(suffix_embeds)
        attention_mask.append(suffix_attns)

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)

        return self.llm_model.generate(inputs_embeds=inputs_embeds,
                                       attention_mask=attention_mask,
                                       generation_config=generation_config)

    @torch.no_grad()
    def chat(
        self,
        history,
        generation_config=None,
        speech_llm_input_ids=None,
        max_speech_unit_length=30 * 50,
        is_speech_generate_run=True,
        enable_thinking=False,
    ):
        inputs_embeds = []

        multi_turn_indices = []
        index = 0
        text_llm_prefix_len = 0
        speech_llm_prefix_embeds = speech_llm_prefix_attention_mask = None
        print('********************* history start ***************************')
        for h in history:
            if len(h) == 1:
                # text
                input_ids = h[0]
                embeds = self.llm_model.get_input_embeddings()(input_ids)
                inputs_embeds.append(embeds)
                multi_turn_indices.append(index)
                index += embeds.size(1)
                print(f"text embeds: {embeds.shape}")
                multi_turn_indices.append(index)
                if text_llm_prefix_len == 0 and speech_llm_input_ids is not None:
                    text_llm_prefix_len = input_ids.size(-1)
                    speech_llm_prefix_embeds = self.llm_model.get_input_embeddings(
                    )(speech_llm_input_ids)
                    speech_llm_prefix_attention_mask = torch.ones(
                        speech_llm_prefix_embeds.size(0),
                        speech_llm_prefix_embeds.size(1),
                        dtype=torch.long).to(speech_llm_prefix_embeds.device)
            elif len(h) == 2:
                # speech
                speech_values, speech_attention_mask = h[0], h[1]
                speech_embeds, _ = self.get_speech_features(
                    speech_values, speech_attention_mask)
                inputs_embeds.append(speech_embeds)
                print(f"speech_embeds: {speech_embeds.shape}")
                index += speech_embeds.size(1)
            else:
                raise NotImplementedError
        print('********************* history end ***************************')
        # multi_turn_for_decoder_indices = [
        #     [0, multi_turn_indices[1] if speech_llm_input_ids is None else speech_llm_prefix_embeds.size(1)],
        #     [multi_turn_indices[-4], -1]]
        # print(f"multi_turn_for_decoder_indices: {multi_turn_for_decoder_indices}")
        # print(generation_config)
        # print(f"inputs_embeds: {len(inputs_embeds)}, {inputs_embeds[0].shape}")
        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        multi_turn_for_decoder_indices = [
            [
                0,
                multi_turn_indices[1]
                if speech_llm_input_ids is None
                else speech_llm_prefix_embeds.size(1),
            ],
            [multi_turn_indices[-4] - int(inputs_embeds.size(1)), -1],
        ]
        # inputs_embeds = inputs_embeds[-1]
        print(f"inputs_embeds: {inputs_embeds.shape}")
        print(f"multi_turn_for_decoder_indices: {multi_turn_for_decoder_indices}")
        print(f"text_llm_prefix_len: {text_llm_prefix_len}")
        print(
            f"speech_llm_prefix_len: {speech_llm_prefix_embeds.size(1) if speech_llm_input_ids is not None else 0}"
        )

        custom_params = {
            "speech_llm_prefix_embeds": speech_llm_prefix_embeds,
            "speech_llm_prefix_attention_mask":
            speech_llm_prefix_attention_mask,
            "text_llm_prefix_len": text_llm_prefix_len,
            "text_label": None,
            "is_teacher_forcing": False,
            "prompt_unit": None,
            'max_speech_unit_length': max_speech_unit_length,
            'is_speech_generate_run': is_speech_generate_run,
            'multi_turn_for_decoder_indices': multi_turn_for_decoder_indices,
            'enable_thinking': enable_thinking
        }

        outputs = self.llm_model.generate(inputs_embeds=inputs_embeds,
                                          generation_config=generation_config,
                                          custom_params=custom_params)

        return outputs.sequences, outputs.speech_units
