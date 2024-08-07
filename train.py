# Copyright (c) 2024 Binbin Zhang(binbzha@qq.com)
# This code is based on the QWen2 from
# https://github.com/QwenLM/Qwen2/blob/main/examples/sft/finetune.py

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import safetensors
import torch
import torchaudio
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          deepspeed, Trainer)
from transformers.trainer_pt_utils import LabelSmoother
import whisper


@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-7B")
    whisper_model_name_or_path: Optional[str] = field(default="tiny")
    encoder_ds_rate: int = 2
    encoder_projector_ds_rate: int = 5
    projector_hidden_size: int = 2048
    projector_model_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."})
    test_data_path: str = field(default=None,
                                metadata={"help": "Path to the test data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adafactor")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length"},
    )


class SpeechDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int = 512,
        inference: bool = False,
    ):
        super(SpeechDataset, self).__init__()
        print("Formatting inputs...")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inference = inference
        self.raw_data = []
        with open(data_path, "r") as f:
            for line in f:
                self.raw_data.append(json.loads(line))

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        msg = self.raw_data[i]
        # load audio and pad/trim it to fit 30 seconds
        speech_len = 300
        audio, sample_rate = torchaudio.load(msg['wav'])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        audio = audio[0]  # get the first channel
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        ids_audio = [0] * int(mel.shape[1] / 10)  # 10x downsample
        tgt_audio = [IGNORE_TOKEN_ID] * len(ids_audio)
        chat = [{"role": "user", "content": "Trascribe the speech"}]
        if self.inference:
            kwargs = {'add_generation_prompt': True}
        else:
            chat.append({"role": "assistant", "content": msg['txt']})
            kwargs = {
                'padding': 'max_length',
                'max_length': self.max_len - speech_len,
                'truncation': True,
                'add_generation_prompt': False,
            }
        ids_text = self.tokenizer.apply_chat_template(chat,
                                                      tokenize=True,
                                                      **kwargs)
        ids = ids_audio + ids_text
        tgt = tgt_audio + ids_text
        input_ids = torch.tensor(ids, dtype=torch.int)
        target_ids = torch.tensor(tgt, dtype=torch.int)
        target_ids[target_ids == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        ret = dict(
            input_ids=input_ids,
            labels=target_ids,
            attention_mask=attention_mask,
            mel=mel,
        )
        return ret


class ProjectorCov1d(nn.Module):

    def __init__(self, config, encoder_dim, llm_dim):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.conv1d = nn.Conv1d(in_channels=encoder_dim,
                                out_channels=encoder_dim,
                                kernel_size=self.k,
                                stride=self.k,
                                padding=0)
        self.linear1 = nn.Linear(encoder_dim, config.projector_hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(config.projector_hidden_size, llm_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


class SpeechLLM(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config,
        llm: nn.Module,
        encoder: nn.Module,
        projector: nn.Module,
    ):
        super().__init__(config)
        self.config = config  # copy llm's config
        self.llm = llm
        self.encoder = encoder
        self.projector = projector
        self._keys_to_ignore_on_save = set()
        # Do not save the parameter of llm and whisper
        for k in self.llm.state_dict().keys():
            self._keys_to_ignore_on_save.add('llm.' + k)
        for k in self.encoder.state_dict().keys():
            self._keys_to_ignore_on_save.add('encoder.' + k)

    def get_input_embedding(self, input_ids, mel):
        # whisper, 30s, 2x downsample = 1500
        speech_size = 300
        speech_emb = self.encoder.embed_audio(mel)  # (b, n_mel, 1500)
        # projector, x 5x downsample = 300
        speech_proj = self.projector(speech_emb)  # (b, x, 300)
        text_emb = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat((speech_proj, text_emb[:, speech_size:, :]),
                                  dim=1)
        return inputs_embeds

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mel: torch.LongTensor = None,
    ):
        inputs_embeds = self.get_input_embedding(input_ids, mel)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mel: torch.LongTensor = None,
        decode_config=None,
    ):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        mel = mel.to(device)
        inputs_embeds = self.get_input_embedding(input_ids, mel)
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            top_p=1.0,
            num_beams=decode_config.num_beams,
            max_new_tokens=decode_config.max_new_tokens,
            eos_token_id=[151643, 151645],
        )
        return model_outputs

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def freeze_encoder(self):
        freeze_model(self.encoder)

    def freeze_llm(self):
        freeze_model(self.llm)

    def load_projector(self, projector_path):
        projector_state_dict = safetensors.torch.load_file(projector_path)
        self.load_state_dict(projector_state_dict, strict=False)


def init_model(model_args):
    encoder = whisper.load_model(model_args.whisper_model_name_or_path)
    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }
    # Load llm model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.llm_model_name_or_path)
    config.use_cache = False
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        config=config,
        device_map=None,
        **model_load_kwargs,
    )
    encoder_dim = encoder.dims.n_audio_state
    llm_dim = config.hidden_size
    projector = ProjectorCov1d(model_args, encoder_dim, llm_dim)
    total_params = sum(p.numel() for p in projector.parameters())
    print('Projector total params: {:.2f}M'.format(total_params / 1024 / 1024))
    model = SpeechLLM(config, llm_model, encoder, projector)
    if model_args.projector_model_path is not None:
        model.load_projector(model_args.projector_model_path)
    return model


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    model = init_model(model_args)
    model.freeze_llm()
    model.freeze_encoder()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )

    print("Loading data...")
    train_dataset = SpeechDataset(data_args.data_path,
                                  tokenizer=tokenizer,
                                  max_len=training_args.model_max_length)
    if data_args.eval_data_path:
        eval_dataset = SpeechDataset(data_args.eval_data_path,
                                     tokenizer=tokenizer,
                                     max_len=training_args.model_max_length)
    else:
        eval_dataset = None
    # Start trainer
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
