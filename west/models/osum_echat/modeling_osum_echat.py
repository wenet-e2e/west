# Copyright (c) 2025 Xuelong Geng(xlgeng@mail.nwpu.edu.cn)

import logging
from typing import Optional

import torch
import wenet
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          GenerationMixin, PreTrainedModel,
                          StoppingCriteriaList)
from wenet.models.transformer.encoder import TransformerEncoder

from .configuration_osum_echat import OSUMEChatConfig
from .cumstom_stop_criteria import (InterruptStopper, MaxTokenStopper,
                                    S2SStopCriteria)


class ProjectorTransformerWithCov1d(nn.Module):

    def __init__(self, encoder_dim, llm_dim):
        super().__init__()
        self.speech_transformer = TransformerEncoder(
            input_size=encoder_dim,
            output_size=encoder_dim,
            attention_heads=4,
            linear_units=2560,
            num_blocks=4,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.0,
            input_layer="linear",
            pos_enc_layer_type="abs_pos",
            normalize_before=True)
        self.conv = torch.nn.Sequential(
            torch.nn.ConstantPad1d((2, 0), 0.0),
            torch.nn.Conv1d(encoder_dim,
                            encoder_dim, 3, 1),
            torch.nn.GELU(),
            torch.nn.ConstantPad1d((2, 0), 0.0),
            torch.nn.Conv1d(encoder_dim,
                            encoder_dim, 3, 2),
            torch.nn.GELU(),
            torch.nn.ConstantPad1d((2, 0), 0.0),
            torch.nn.Conv1d(encoder_dim,
                            encoder_dim, 3, 2),
            torch.nn.GELU(),
        )
        self.speech_llama_proj = nn.Linear(
            encoder_dim, llm_dim)

    def forward(self, encoder_out, encoder_mask):
        encoder_out = encoder_out.transpose(1, 2)
        encoder_out = self.conv(encoder_out)
        conv_out = encoder_out.transpose(1, 2)
        encoder_mask = encoder_mask[:, :, 0::2]
        encoder_mask = encoder_mask[:, :, 0::2]
        conv_downsample_len = encoder_mask.squeeze(
            1).sum(-1)
        speech_embeds, encoder_mask = self.speech_transformer(
            conv_out, conv_downsample_len)
        speech_embeds4llm = self.speech_llama_proj(
            speech_embeds)
        return speech_embeds4llm, encoder_mask.squeeze(
            1)


class OSUMEChat(PreTrainedModel, GenerationMixin):
    model_type = 'osum_echat'
    config_class = OSUMEChatConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: OSUMEChatConfig,
                 *inputs, **kwargs):
        """
        TODO(Xuelong Geng): Complete the design of OSUMEChat
        """
        super().__init__(config, *inputs,
                         **kwargs)
        self.encoder = wenet.load_model(
            config.wenet_model_name_or_path)
        del self.encoder.decoder
        del self.encoder.ctc
        logging.info(
            f'self.encoder: {self.encoder}')
        llm_config = AutoConfig.from_pretrained(
            config.llm_model_name_or_path)
        logging.info(
            f'采用如下 LLM： {config.llm_model_name_or_path}'
        )
        if config.no_init_llm:
            logging.info(
                'No init llm, only load llm structure'
            )
            self.llm = AutoModelForCausalLM.from_config(
                llm_config, )
            self.llm.to(torch.bfloat16)
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name_or_path,
                config=llm_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                # or "flex_attention"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name_or_path,
            use_fast=False,
            trust_remote_code=True)
        self.embed_tokens = self.llm.model.embed_tokens
        logging.info(
            f'self.llm: {self.llm}')
        self.projector = ProjectorTransformerWithCov1d(
            encoder_dim=self.encoder.encoder.
            output_size(),
            llm_dim=llm_config.hidden_size,
        )
        logging.info(
            f'self.projector: {self.projector}')

        self.speech_token_emded = torch.nn.Embedding(
            config.speech_token_num + 2,
            llm_config.hidden_size)
        self.speech_head = torch.nn.Linear(
            llm_config.hidden_size,
            config.speech_token_num)
        self.add_embed_head = True
        self.IGNORE_ID = -100
        self.speech_token_num = config.speech_token_num
        self.init_custom_stop_criteria()

    @torch.autocast(device_type="cuda",
                    dtype=torch.bfloat16)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[
                torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            position_ids: Optional[
                torch.LongTensor] = None,
            audio_offsets: Optional[
                torch.LongTensor] = None,
            audio_features: Optional[
                torch.FloatTensor] = None,
            audio_features_lengths: Optional[
                torch.LongTensor] = None,
            talker_features: Optional[
                torch.FloatTensor] = None,
            talker_features_lengths: Optional[
                torch.LongTensor] = None,
            talker_offsets: Optional[
                torch.LongTensor] = None,
            batch_idx: Optional[
                torch.LongTensor] = None,
            has_audio: Optional[
                torch.BoolTensor] = None,
            **kwargs,
    ):
        """
        TODO(Xuelong Geng): Complete the design of OSUMEChat
        """

    @torch.autocast(device_type="cuda",
                    dtype=torch.bfloat16)
    def generate(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[
                torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            position_ids: Optional[
                torch.LongTensor] = None,
            audio_offsets: Optional[
                torch.LongTensor] = None,
            audio_features: Optional[
                torch.FloatTensor] = None,
            audio_features_lengths: Optional[
                torch.LongTensor] = None,
            batch_idx: Optional[
                torch.LongTensor] = None,
            has_audio: Optional[
                torch.BoolTensor] = None,
            **kwargs,
    ):
        """
        TODO(Xuelong Geng): Complete the design of OSUMEChat
        """
        self.llm.eval()
        self.set_task_type("S2S")
        self.do_add_speech_embed_head()
        # =====================准备input embedding=====================
        encoder_out, encoder_mask = self.encoder._forward_encoder(
            audio_features,
            audio_features_lengths)
        speech_embeds, speech_masks = self.projector(
            encoder_out, encoder_mask)

        speech_embeds, speech_masks, _ = self._add_bos_eos(
            0 + self.speech_token_num, None,
            speech_embeds, speech_masks, None)
        device = speech_embeds.device
        qwen_instruct_prompt_pattern_1 = (
            "<|im_start|>system\nYou are OSUM-chat,"
            " a speech-to-speech dialogue assistant by ASLP Lab. "
            "You understand both the meaning and paralinguistic"
            " cues in speech. Before responding, "
            "first output your reasoning inside "
            "<think>...</think end>, analyzing the user’s "
            "words and vocal cues. Then generate a reply "
            "with appropriate text and emotionally "
            "matched synthetic speech.<|im_end|>\n<|im_start|>user\n"
        )
        prompt_pattern1 = self.tokenizer(
            [qwen_instruct_prompt_pattern_1] *
            len(audio_features),
            return_tensors="pt")['input_ids'].to(
            speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(
            prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer(
            [qwen_instruct_prompt_pattern_2] *
            len(audio_features),
            return_tensors="pt")['input_ids'].to(
            speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(
            prompt_pattern2)

        hyps = [4098]
        token_emb = self.speech_token_emded(
            torch.tensor(hyps[-1:]).to(
                device)).unsqueeze(0)

        embeds = torch.cat([
            prompt_pattern1_embeds, speech_embeds,
            token_emb, prompt_pattern2_embeds
        ],
            dim=1)
        if self.embed_tokens.weight.dtype == torch.bfloat16:
            embeds = embeds.to(torch.bfloat16)

        top_k = 10
        top_p = 0.9
        temperature = 1.2
        invalid_eos = 10000000
        llm_out = self.llm.generate(
            inputs_embeds=embeds,
            max_new_tokens=2000,
            eos_token_id=invalid_eos,
            cache_implementation="static",
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stopping_criteria=self.
            s2s_stop_criteria,
            do_compile=True,
            repetition_penalty=1.0,
        )

        text_eos_idx = (
                llm_out[0] == 151645).nonzero(
            as_tuple=True)[0][0].item()
        text_res = llm_out[:, :text_eos_idx - 1]
        speech_res = llm_out[:, text_eos_idx + 1:-1]

        output_text = self.tokenizer.batch_decode(
            text_res,
            add_special_tokens=False,
            skip_special_tokens=True)
        return (output_text, text_res, speech_res)

    def init_tokenizer(self):
        """
        TODO(Xuelong Geng): Complete the design of OSUMEChat
        """

    def init_custom_stop_criteria(self):
        """
        创建需要的stop criteria
        1. 对于t2t任务，遇到text_eos停止
        2. 对于t2s任务，遇到speech_eos停止
        3. 对于s2s任务，遇到speech_eos停止
        同时要取消原本的停止条件
        if generation_config._eos_token_tensor is not None:
        取消 generation_config._eos_token_tensor 的停止，尝试直接给一个大于vocb_size的eos_token
        """
        self.interrupt = InterruptStopper()
        self.s2s_stop_criteria = StoppingCriteriaList(
        )
        self.s2s_stop_criteria.append(
            S2SStopCriteria(text_eos_id=151645,
                            speech_eos_id=self.
                            speech_token_num - 1))
        self.s2s_stop_criteria.append(
            MaxTokenStopper(2000))
        self.s2s_stop_criteria.append(
            self.interrupt)

    def set_task_type(self, task_type: str):
        """设置任务类型，用于设置生成的初始类型
        Args:
            task_type (str): 任务类型，从("ASR", "TTS", "S2S")选择
            ASR: 语音到文本范式
            TTS: 文本到语音范式
            S2S: 语音到语音范式
        """
        assert task_type in ("ASR", "TTS", "S2S")
        if task_type == "ASR":
            self.llm.text_phase = True
        elif task_type == "TTS":
            self.llm.text_phase = False
        elif task_type == "S2S":
            self.llm.text_phase = True

    def do_add_speech_embed_head(self):
        if self.add_embed_head:
            self.llm.speech_token_emded = self.speech_token_emded.to(
                torch.bfloat16)
            self.llm.speech_head = self.speech_head.to(
                torch.bfloat16)
            self.add_embed_head = False

    def _add_bos_eos(self,
                     bos,
                     eos,
                     inputs_embeds,
                     attention_mask,
                     target=None):
        B = len(inputs_embeds)
        bos_eos_target = torch.full(
            [B, 1], self.IGNORE_ID).to(
            inputs_embeds.device)  # B,1
        bos_eos_mask = torch.full(
            [B, 1],
            True).to(inputs_embeds.device)  # B, 1

        if bos is not None:
            bos_embed = self.speech_token_emded(
                torch.full([B, 1], bos).to(
                    inputs_embeds.device)
            )  # B, 1, D
            inputs_embeds = torch.cat(
                (bos_embed, inputs_embeds),
                1)  # B, (1+T), D
            attention_mask = torch.cat(
                (bos_eos_mask, attention_mask),
                1)  # B, (1+T)
            if target is not None:
                target = torch.cat(
                    (bos_eos_target, target),
                    1)  # B, (1+T), D

        if eos is not None:
            eos_embed = self.speech_token_emded(
                torch.full([B, 1], eos).to(
                    inputs_embeds.device)
            )  # B, 1, D
            inputs_embeds = torch.cat(
                (inputs_embeds, eos_embed),
                1)  # B, (1+T+1), D
            attention_mask = torch.cat(
                (attention_mask, bos_eos_mask),
                1)  # B, (1+T+1)
            if target is not None:
                target = torch.cat(
                    (target, bos_eos_target),
                    1)  # B, (1+T+1), D

        return inputs_embeds, attention_mask, target
