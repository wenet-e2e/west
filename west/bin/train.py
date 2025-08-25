# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
# This code is based on the QWen2 from
# https://github.com/QwenLM/Qwen2/blob/main/examples/sft/finetune.py

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Union

import torch
from torch import nn
from transformers import (AutoConfig, AutoModel, HfArgumentParser, Trainer,
                          TrainerCallback, TrainingArguments)

from west.dataset.dataset import DataArguments, SpeechDataset
from west.dataset.extractor import Extractor


@dataclass
class TrainingArguments(TrainingArguments):
    optim: str = field(default="adafactor")
    model_config_path: str = field(default='')


class MyTrainer(Trainer):

    def training_step(self,
                      model: nn.Module,
                      inputs: dict[str, Union[torch.Tensor, Any]],
                      num_items_in_batch=None) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        # loss compute will aplly `shift_labels`
        # See https://github.com/huggingface/transformers/blob/v4.52.2/src/transformers/trainer.py#L3825  # noqa
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model,
                                              inputs,
                                              return_outputs=True)
        # Compute accuracy (ignoring ignore_index)
        if "labels" in inputs:
            logits = outputs.logits[..., :-1, :].contiguous()
            labels = inputs["labels"][..., 1:].contiguous()
            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            correct = (preds == labels) & mask
            accuracy = correct.sum().item() / mask.sum().item()
            interval = \
                self.args.logging_steps * self.args.gradient_accumulation_steps
            if self.state.global_step % interval == 0:
                self.log({
                    "train_loss": loss.detach().cpu().item(),
                    "train_accuracy": accuracy
                })

        if self.args.n_gpu > 1:
            loss = loss.mean()
        loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        return loss.detach()


class ProfilerCallback(TrainerCallback):

    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        self.profiler = None
        self.started = False
        self.global_step = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.global_step = state.global_step

        if not self.started:
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=999,
                                                 warmup=0,
                                                 active=1,
                                                 repeat=0,
                                                 skip_first=5),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.log_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_modules=True)
            self.profiler.__enter__()
            self.started = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.started and self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    config = AutoConfig.from_pretrained(training_args.model_config_path)
    model = AutoModel.from_config(config)
    tokenizer = model.init_tokenizer()
    extractor = Extractor.get_class(model.model_type)(tokenizer)

    print("Loading data...")
    train_dataset = SpeechDataset(extractor, data_args)
    # Start trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda x: x[0],
        callbacks=[ProfilerCallback(log_dir=training_args.logging_dir)],
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
