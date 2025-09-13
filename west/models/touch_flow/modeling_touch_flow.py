# Copyright (c) 2024 Tsinghua Univ. (authors: Xingchen Song)
#               2025 Binbin Zhang(binbzha@qq.com)

import math
import random
from typing import Optional

import s3tokenizer
import torch
import torch.nn.functional as F
import wespeaker
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)

from west.utils.mask import make_pad_mask, non_causal_mask
from west.utils.utils import freeze_module

from .configuration_touch_flow import TouchFlowConfig


class SinusoidalPosEmb(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() *
                        -emb)  # (half_dim,)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # (B, dim)
        return emb


class TouchFlow(PreTrainedModel):
    """flow model based on huggingface transformers"""
    model_type = 'touch_flow'
    config_class = TouchFlowConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: TouchFlowConfig):
        super().__init__(config)
        llm_config = AutoConfig.from_pretrained(config.llm_model_name_or_path)
        # Load llm model and tokenizer
        self.llm = AutoModelForCausalLM.from_config(config=llm_config)
        config.hidden_size = llm_config.hidden_size
        speech_tokenizer = s3tokenizer.load_model(
            'speech_tokenizer_v1_25hz', config.s3tokenizer_model_name_or_path)
        self.speech_tokenizer = speech_tokenizer
        speaker_model = wespeaker.load_model_pt(config.speaker_model_path)
        self.speaker_model = speaker_model
        freeze_module(self.speech_tokenizer)
        freeze_module(self.speaker_model)
        self.vocab_size = self.llm.vocab_size
        mel_dim = 80
        hidden_size = config.hidden_size
        self.spk_encoder = torch.nn.Linear(192, mel_dim)
        self.time_embeddings = SinusoidalPosEmb(hidden_size)
        self.time_encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size * 4, mel_dim),
        )
        self.token_encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size * 4, mel_dim),
        )
        self.input_projector = torch.nn.Linear(mel_dim * 5, hidden_size)
        self.mel_projector = torch.nn.Linear(hidden_size, mel_dim)

    def tie_weights(self):
        return self.llm.tie_weights()

    def interpolate(self, x, ylens=None):
        # x in (B, T, D)
        mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(),
                          size=ylens.max(),
                          mode='linear')
        out = x.transpose(1, 2).contiguous()
        return out * mask, ylens

    def forward(
        self,
        mel_speaker: Optional[torch.FloatTensor] = None,
        mel_speaker_lengths: Optional[torch.LongTensor] = None,
        mel_token: Optional[torch.FloatTensor] = None,
        mel_token_lengths: Optional[torch.LongTensor] = None,
        mel_vocoder: Optional[torch.FloatTensor] = None,
        mel_vocoder_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """ All mel_* tensors are in (B, T, D)
        """
        device = mel_speaker.device
        self.speech_tokenizer.eval()
        self.speaker_model.eval()
        B, T, M = mel_vocoder.shape
        # Condition speech token, compute speech token on-the-fly
        speech_token, speech_token_lengths = self.speech_tokenizer.quantize(
            mel_token.transpose(1, 2), mel_token_lengths)
        speech_token = speech_token.clone()
        emb = self.token_encoder(self.llm.model.embed_tokens(speech_token))
        mask = ~make_pad_mask(speech_token_lengths).to(device)
        emb = emb * mask.unsqueeze(-1)
        token_cond, _ = self.interpolate(emb, mel_vocoder_lengths)
        # Condition speaker embedding, compute speaker embedding on-the-fly
        # Use the min length in batch to compute embedding for each item
        spk_emb = self.speaker_model(mel_speaker)
        spk_cond = self.spk_encoder(F.normalize(spk_emb, dim=1))
        spk_cond = spk_cond.unsqueeze(1).repeat(1, T, 1)  # (B, T, M)
        # Condition mel prompt, sample at the begining in traning, and
        # we can use prompt speech as condition in inference.
        mel_cond = torch.zeros(mel_vocoder.shape, device=device)  # (B, T, M)
        for i, j in enumerate(mel_vocoder_lengths):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            mel_cond[i, :index] = mel_vocoder[i, :index]
        # Condition randome timestep
        t = torch.rand([B, 1, 1], device=device, dtype=mel_vocoder.dtype)
        if self.config.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        t_cond = self.time_encoder(
            self.time_embeddings(t.squeeze()).to(t.dtype))  # (B, M)
        t_cond = t_cond.unsqueeze(1).repeat(1, T, 1)  # (B, T, M)
        # during training, we randomly drop condition to trade off model
        # coverage and sample fidelity.
        # cfg is short for `Classifier-Free Guidance`
        if self.config.training_cfg_rate > 0:
            cfg_mask = torch.rand(B,
                                  device=device) > self.config.training_cfg_rate
            spk_cond = spk_cond * cfg_mask.view(-1, 1, 1)
            token_cond = token_cond * cfg_mask.view(-1, 1, 1)
            mel_cond = mel_cond * cfg_mask.view(-1, 1, 1)
        # See name & details in `FLOW MATCHING FOR GENERATIVE MODELING`
        # in `https://arxiv.org/abs/2210.02747`
        p0 = torch.randn_like(mel_vocoder)  # random noise
        p1 = mel_vocoder
        pt = (1 - (1 - self.config.sigma_min) * t) * p0 + t * p1
        ut = p1 - (1 - self.config.sigma_min) * p0
        inputs = torch.cat([pt, token_cond, t_cond, spk_cond, mel_cond],
                           dim=-1)  # (B, T, 5*M)
        inputs = self.input_projector(inputs)  # (B, T, D)
        mask = ~make_pad_mask(mel_vocoder_lengths).to(device)  # (B, T)
        att_mask = non_causal_mask(mel_vocoder_lengths).to(device)  # (B, T, T)
        att_mask = att_mask.unsqueeze(1).float()  # (B, 1, T, T)
        result = self.llm.model(inputs_embeds=inputs,
                                attention_mask=att_mask,
                                return_dict=True)
        vt = self.mel_projector(result.last_hidden_state)  # (B, T, M)
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        loss = F.mse_loss(vt * mask, ut * mask,
                          reduction="sum") / (torch.sum(mask) * M)
        return {'loss': loss}

    # @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def inference(
        self,
        mel_speaker: Optional[torch.FloatTensor] = None,
        mel_speaker_lengths: Optional[torch.LongTensor] = None,
        mel_token: Optional[torch.FloatTensor] = None,
        mel_token_lengths: Optional[torch.LongTensor] = None,
        mel_vocoder: Optional[torch.FloatTensor] = None,
        mel_vocoder_lengths: Optional[torch.LongTensor] = None,
        llm_token: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Args:
            llm_token: speech token predicted by LLM
        """
        device = mel_speaker.device
        B, _, M = mel_vocoder.shape
        assert (B == 1)
        # Condition speech token, compute speech token on-the-fly
        prompt_token, prompt_token_lengths = self.speech_tokenizer.quantize(
            mel_token.transpose(1, 2), mel_token_lengths)
        mel_len1 = mel_vocoder.shape[1]
        mel_len2 = int(llm_token.shape[1] / 25 * 22050 / 256)
        T = mel_len1 + mel_len2
        speech_token = torch.concat([prompt_token, llm_token], dim=1)

        emb = self.token_encoder(self.llm.model.embed_tokens(speech_token[0]))
        output_length = torch.tensor([T], dtype=torch.long, device=device)
        token_cond, _ = self.interpolate(emb.unsqueeze(0), output_length)
        # Condition speaker embedding, compute speaker embedding on-the-fly
        spk_emb = self.speaker_model(mel_speaker)
        spk_cond = self.spk_encoder(F.normalize(spk_emb, dim=1))
        spk_cond = spk_cond.unsqueeze(1).repeat(1, T, 1)  # (B, T, M)
        # Condition mel prompt
        mel_cond = torch.zeros((B, T, M), device=device)  # (B, T, M)
        mel_cond[:, :mel_len1, :] = mel_vocoder
        # Condition t
        t_span = torch.linspace(0,
                                1,
                                self.config.n_timesteps + 1,
                                device=device,
                                dtype=torch.float32)  # (n_timesteps+1, )
        # Sample first noise, pt = p0 = noise
        pt = torch.randn_like(token_cond)
        if self.config.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t, dt = t_span[0], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)  # (B, 1)
        x_in = torch.zeros((2, T, 5 * M), dtype=torch.float, device=device)
        x_in[0:1, :, M:2 * M] = token_cond
        x_in[0:1, :, 3 * M:4 * M] = spk_cond
        x_in[0:1, :, 4 * M:5 * M] = mel_cond
        vocoder_lengths = torch.tensor([T], dtype=torch.long, device=device)
        att_mask = non_causal_mask(vocoder_lengths).to(device)  # (B, T, T)
        att_mask = att_mask.unsqueeze(1).float()  # (B, 1, T, T)
        for step in range(1, len(t_span)):
            x_in[:, :, 0:M] = pt
            t_cond = self.time_encoder(
                self.time_embeddings(t.squeeze()).to(t.dtype))
            t_cond = t_cond.unsqueeze(1).repeat(1, T, 1)  # (1, T, M)
            x_in[:, :, 2 * M:3 * M] = t_cond
            inputs = self.input_projector(x_in)  # (2, T, D)
            result = self.llm.model(inputs_embeds=inputs,
                                    attention_mask=att_mask,
                                    return_dict=True)
            vt = self.mel_projector(result.last_hidden_state)  # (2, T, M)
            alpha = self.config.inference_cfg_rate
            # classifier free guidance (CFG) inference, see paper
            # Voicebox: Text-Guided Multilingual Universal Speech Generation
            # at Scale, https://arxiv.org/abs/2306.15687
            vt = (1.0 + alpha) * vt[:1] - alpha * vt[1:2]
            pt = pt + dt * vt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return pt[:, mel_len1:, :]

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model_name_or_path)
        if 'Qwen' in self.config.llm_model_name_or_path:
            tokenizer.bos_token = tokenizer.eos_token
        return tokenizer
