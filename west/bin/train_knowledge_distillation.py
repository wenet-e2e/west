"""On-Policy Knowledge Distillation training script for Audio LLM models.

This script implements on-policy knowledge distillation where:
1. Student model generates completions
2. Teacher model evaluates student's completions
3. Student learns to match teacher's distribution
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import (AutoProcessor, HfArgumentParser,
                          Qwen2_5OmniForConditionalGeneration,
                          Qwen2AudioForConditionalGeneration,
                          TrainingArguments)

from west.dataset.hf_dataset import HFAudioDataset
from west.trainer.kd_trainer import (KnowledgeDistillationTrainer,
                                     RemoteKnowledgeDistillationTrainer)
from west.utils.constants import TEMPLATE_MAP
from west.utils.rewards import (accuracy_reward, format_reward,
                                format_reward_answer, format_reward_think)


def is_url(path: str) -> bool:
    """Check if the path is a URL."""
    return path.startswith("http://") or path.startswith("https://")


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Arguments for On-Policy Knowledge Distillation training."""
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Learning rate"},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed config path"},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model name or path"},
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher model name or path"},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for model"},
    )
    hf_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to HuggingFace dataset"},
    )
    use_wandb: Optional[str] = field(
        default="false",
        metadata={"help": "Whether to use wandb for logging"},
    )
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to False to keep all columns"},
    )

    # Training
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for training"},
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "Number of gradient accumulation steps"},
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Number of training epochs"},
    )
    max_steps: int = field(
        default=1000,
        metadata={"help": "Maximum number of training steps"},
    )

    # Generation
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Maximum prompt length for generation"},
    )
    max_completion_length: int = field(
        default=4096,
        metadata={"help": "Maximum completion length for generation"},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature"},
    )
    topk_logits_k: Optional[int] = field(
        default=64,
        metadata={"help": "Top-k logits for distillation. None = full vocabulary, 64 is a common choice"},
    )

    # Template
    template: str = field(
        default="default",
        metadata={"help": "Prompt template type: default, think, new"},
    )

    # Logging & Saving
    logging_steps: int = field(
        default=1,
        metadata={"help": "Log every N steps"},
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save every N steps"},
    )
    save_only_model: bool = field(
        default=True,
        metadata={"help": "Save only model weights"},
    )
    report_to: list[str] = field(
        default_factory=list,
        metadata={"help": "Reporting integrations"},
    )
    run_name: str = field(
        default="on_policy_distillation",
        metadata={"help": "Run name for logging"},
    )

    # Misc
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"},
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Data shuffling seed"},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 precision"},
    )
    num_generations: int = field(
        default=1,
        metadata={"help": "Number of generations per prompt for rollout"},
    )


def main():
    parser = HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if not args.report_to:
        args.report_to = ["wandb"] if args.use_wandb == "true" else []

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # logging.disable(logging.WARNING)
    logging.info(f"Training arguments: {args}")

    logging.info(f"Loading model from: {args.model_name_or_path}")
    if "Qwen2-Audio-7B-Instruct" in args.model_name_or_path:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_name_or_path)
    elif "Qwen2.5-Omni" in args.model_name_or_path:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            enable_audio_output=False,
        )
    else:
        raise ValueError(f"Model {args.model_name_or_path} not supported")

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    # Check if teacher is a remote API or local model
    use_remote_teacher = is_url(args.teacher_model_name_or_path)

    if use_remote_teacher:
        logging.info(f"Using remote teacher API: {args.teacher_model_name_or_path}")
        teacher_model = None
        teacher_processor = None
    else:
        logging.info(f"Loading local teacher model from: {args.teacher_model_name_or_path}")
        if "Qwen2-Audio-7B-Instruct" in args.teacher_model_name_or_path:
            teacher_model = Qwen2AudioForConditionalGeneration.from_pretrained(args.teacher_model_name_or_path)
        elif "Qwen2.5-Omni" in args.teacher_model_name_or_path or "omni" in args.teacher_model_name_or_path:
            teacher_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                args.teacher_model_name_or_path,
                enable_audio_output=False,
            )
        else:
            raise ValueError(f"Teacher model {args.teacher_model_name_or_path} not supported")
        teacher_processor = AutoProcessor.from_pretrained(args.teacher_model_name_or_path)

    prompt_template = TEMPLATE_MAP[args.template]
    logging.info(f"Using template: {args.template}")
    train_dataset = HFAudioDataset(
        args.hf_dataset_path,
        processor=processor,
        split="train",
        max_prompt_length=args.max_prompt_length,
        prompt_template=prompt_template,
    )
    eval_dataset = HFAudioDataset(
        args.hf_dataset_path,
        processor=processor,
        split="validation",
        max_prompt_length=args.max_prompt_length,
        prompt_template=prompt_template,
    )
    # Reward functions for monitoring (not used in loss, only for logging)
    if args.template == "default":
        reward_funcs = [accuracy_reward, format_reward]
    elif args.template == "think":
        reward_funcs = [accuracy_reward, format_reward_answer, format_reward_think]
    elif args.template == "caption":
        reward_funcs = None
    else:
        raise ValueError(f"Template {args.template} not supported")

    if use_remote_teacher:
        trainer = RemoteKnowledgeDistillationTrainer(
            model=model,
            teacher_api_base=args.teacher_model_name_or_path,
            processor=processor,
            args=args,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=train_dataset.collate_fn,
        )
    else:
        trainer = KnowledgeDistillationTrainer(
            model=model,
            teacher_model=teacher_model,
            processor=processor,
            teacher_model_processor=teacher_processor,
            args=args,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=train_dataset.collate_fn,
        )

    trainer.train()


if __name__ == "__main__":
    main()
