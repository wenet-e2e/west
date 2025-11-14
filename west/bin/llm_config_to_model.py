# Copyright (c) 2025 Changwei Ma(chwma0@aliyun.com)

import os
from dataclasses import dataclass, field

from transformers import AutoConfig, AutoModelForCausalLM, HfArgumentParser


@dataclass
class CustomModelArguments:
    llm_model_config_path: str = field(
        default='',
        metadata={"help": "Path to LLM model config file or directory"})
    save_model_dir: str = field(
        default='./',
        metadata={"help": "Directory to save the initialized model"})


def main():
    parser = HfArgumentParser(CustomModelArguments)
    custom_args = parser.parse_args_into_dataclasses()[0]
    os.makedirs(custom_args.save_model_dir, exist_ok=True)

    # Initialized pretrained model from config
    llm_config = AutoConfig.from_pretrained(custom_args.llm_model_config_path)
    model = AutoModelForCausalLM.from_config(config=llm_config)

    # Save the initialized model
    model.save_pretrained(custom_args.save_model_dir)
    print(f"Model initialized and saved to: {custom_args.save_model_dir}")


if __name__ == "__main__":
    main()
