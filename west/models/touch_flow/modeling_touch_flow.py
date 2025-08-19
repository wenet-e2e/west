# Copyright (c) 2024 Tsinghua Univ. (authors: Xingchen Song)
#               2025 Binbin Zhang(binbzha@qq.com)

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import s3tokenizer
import safetensors
import torch
import torch.nn.functional as F
import wespeaker
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)

from west.models.model import Model, ModelArgs
from west.utils.mask import make_pad_mask, non_causal_mask

from .length_regulator import InterpolateRegulator


@ModelArgs.register
@dataclass
class TouchFlowArgs:
    s3tokenizer_model_name_or_path: Optional[str] = "speech_tokenizer_v1_25hz"
    speaker_model_path: Optional[str] = ""
    text_tokenizer_path: Optional[str] = ""
    flow_llm_config_path: Optional[str] = field(default="Qwen/Qwen2-7B")
    num_speech_tokens: int = 4096
    flow_model_path: Optional[str] = field(default='')
    t_scheduler: Optional[str] = field(default="cosine")
    sigma_min: float = 1e-6
    training_cfg_rate: float = 0.2
    inference_cfg_rate: float = 0.7
    n_timesteps: int = 5


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


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class TouchFlow(PreTrainedModel, Model):
    """flow model based on huggingface transformers"""
    model_type = 'touch_flow'
    supports_gradient_checkpointing = True

    def __init__(self, args):
        config = AutoConfig.from_pretrained(args.flow_llm_config_path)
        super().__init__(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        speech_tokenizer = s3tokenizer.load_model(
            'speech_tokenizer_v1_25hz', args.s3tokenizer_model_name_or_path)
        self.speech_tokenizer = speech_tokenizer.to(device)
        speaker_model = wespeaker.load_model_local(
            args.speaker_model_path).model
        self.speaker_model = speaker_model.to(device)
        # Load llm model and tokenizer
        self.llm = AutoModelForCausalLM.from_config(config=config)
        self._keys_to_ignore_on_save = set()
        for k in self.speech_tokenizer.state_dict().keys():
            self._keys_to_ignore_on_save.add('speech_tokenizer.' + k)
        for k in self.speaker_model.state_dict().keys():
            self._keys_to_ignore_on_save.add('speaker_model.' + k)
        freeze_model(self.speech_tokenizer)
        freeze_model(self.speaker_model)
        self.vocab_size = self.llm.vocab_size
        self.length_regulator = InterpolateRegulator()
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
        self.args = args
        if args.flow_model_path:
            state_dict = safetensors.torch.load_file(args.flow_model_path)
            self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        mel_speaker: Optional[torch.FloatTensor] = None,
        mel_speaker_lengths: Optional[torch.LongTensor] = None,
        mel_token: Optional[torch.FloatTensor] = None,
        mel_token_lengths: Optional[torch.LongTensor] = None,
        mel_vocoder: Optional[torch.FloatTensor] = None,
        mel_vocoder_lengths: Optional[torch.LongTensor] = None,
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
        # token_emb = self.llm.model.embed_tokens(speech_token)
        # token_cond = self.token_encoder(token_emb)
        # token_cond = F.interpolate(token_cond.transpose(1, 2),
        #                            size=T,
        #                            mode='nearest')
        # token_cond = token_cond.transpose(1, 2)  # (B, T, M)
        speech_token = unpad_sequence(speech_token,
                                      speech_token_lengths,
                                      batch_first=True)
        token_cond = []
        for i, y in enumerate(speech_token):
            emb = self.token_encoder(self.llm.model.embed_tokens(y))
            emb, _ = self.length_regulator(emb.unsqueeze(0),
                                           mel_vocoder_lengths[i].unsqueeze(0))
            token_cond.append(emb.squeeze(0))
        token_cond = pad_sequence(token_cond,
                                  batch_first=True,
                                  padding_value=0.0).to(device)  # (B, T, M)
        # Condition speaker embedding, compute speaker embedding on-the-fly
        # Use the min length in batch to compute embedding for each item
        # min_length = torch.min(mel_speaker_lengths)
        # spk_emb = self.speaker_model(mel_speaker[:, :min_length, :])
        spk_emb = self.speaker_model(mel_speaker)
        spk_cond = self.spk_encoder(F.normalize(spk_emb, dim=1))
        # spk_cond = self.spk_encoder(spk_emb)
        spk_cond = spk_cond.unsqueeze(1).repeat(1, T, 1)  # (B, T, M)
        # spk_cond = torch.zeros((B, 80), dtype=torch.float, device=device)
        # for i in range(B):
        #     m = mel_speaker[i, :mel_speaker_lengths[i],:].unsqueeze(0)
        #     m = self.speaker_model(m)
        #     spk_cond[i] = self.spk_encoder(F.normalize(m, dim=1)).squeeze(0)
        # spk_cond = spk_cond.unsqueeze(1).repeat(1, T, 1)  # (B, T, M)
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
        if self.args.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        t_cond = self.time_encoder(
            self.time_embeddings(t.squeeze()).to(t.dtype))  # (B, M)
        t_cond = t_cond.unsqueeze(1).repeat(1, T, 1)  # (B, T, M)
        # during training, we randomly drop condition to trade off model
        # coverage and sample fidelity.
        # cfg is short for `Classifier-Free Guidance`
        if self.args.training_cfg_rate > 0:
            cfg_mask = torch.rand(B,
                                  device=device) > self.args.training_cfg_rate
            spk_cond = spk_cond * cfg_mask.view(-1, 1, 1)
            token_cond = token_cond * cfg_mask.view(-1, 1, 1)
            mel_cond = mel_cond * cfg_mask.view(-1, 1, 1)
        # See name & details in `FLOW MATCHING FOR GENERATIVE MODELING`
        # in `https://arxiv.org/abs/2210.02747`
        p0 = torch.randn_like(mel_vocoder)  # random noise
        p1 = mel_vocoder
        pt = (1 - (1 - self.args.sigma_min) * t) * p0 + t * p1
        ut = p1 - (1 - self.args.sigma_min) * p0
        inputs = torch.cat([pt, token_cond, t_cond, spk_cond, mel_cond],
                           dim=-1)  # (B, T, 5*M)
        inputs = self.input_projector(inputs)  # (B, T, D)
        mask = ~make_pad_mask(mel_vocoder_lengths).to(device)  # (B, T)
        att_mask = non_causal_mask(mel_vocoder_lengths).to(device)  # (B, T, T)
        att_mask = att_mask.unsqueeze(1)  # (B, 1, T, T)
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

        # token_emb = self.llm.model.embed_tokens(speech_token)
        # token_cond = self.token_encoder(token_emb)
        # token_cond = F.interpolate(token_cond.transpose(1, 2),
        #                            size=T,
        #                            mode='nearest')
        # token_cond = token_cond.transpose(1, 2)  # (B, T, M)

        emb = self.token_encoder(self.llm.model.embed_tokens(speech_token[0]))
        output_length = torch.tensor([T], dtype=torch.long, device=device)
        token_cond, _ = self.length_regulator(emb.unsqueeze(0), output_length)
        # Condition speaker embedding, compute speaker embedding on-the-fly
        spk_emb = self.speaker_model(mel_speaker)
        # spk_cond = self.spk_encoder(spk_emb)
        spk_cond = self.spk_encoder(F.normalize(spk_emb, dim=1))
        spk_cond = spk_cond.unsqueeze(1).repeat(1, T, 1)  # (B, T, M)
        # Condition mel prompt
        mel_cond = torch.zeros((B, T, M), device=device)  # (B, T, M)
        mel_cond[:, :mel_len1, :] = mel_vocoder
        # Condition t
        t_span = torch.linspace(0,
                                1,
                                self.args.n_timesteps + 1,
                                device=device,
                                dtype=torch.float32)  # (n_timesteps+1, )
        # Sample first noise, pt = p0 = noise
        pt = torch.randn_like(token_cond)
        if self.args.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t, dt = t_span[0], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)  # (B, 1)
        x_in = torch.zeros((2, T, 5 * M), dtype=torch.float, device=device)
        x_in[0:1, :, M:2 * M] = token_cond
        x_in[0:1, :, 3 * M:4 * M] = spk_cond
        x_in[0:1, :, 4 * M:5 * M] = mel_cond
        vocoder_lengths = torch.tensor([T], dtype=torch.long, device=device)
        att_mask = non_causal_mask(vocoder_lengths).to(device)  # (B, T, T)
        att_mask = att_mask.unsqueeze(1)  # (B, 1, T, T)
        for step in range(1, self.args.n_timesteps):
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
            alpha = self.args.inference_cfg_rate
            # classifier free guidance (CFG) inference, see paper
            # Voicebox: Text-Guided Multilingual Universal Speech Generation
            # at Scale, https://arxiv.org/abs/2306.15687
            vt = (1.0 + alpha) * vt[:1] - alpha * vt[1:2]
            pt = pt + dt * vt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return pt[:, mel_len1:, :]

    @staticmethod
    def init_tokenizer(args):
        tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer_path)
        if 'Qwen' in args.llm_model_name_or_path:
            tokenizer.bos_token = tokenizer.eos_token
        return tokenizer
