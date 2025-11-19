# Copyright 2025 Chengdong Liang(liangchengdongd@qq.com)

import argparse
import json
import os

import torch
from transformers import AutoConfig, AutoModel

import west  # for init touchasu model  # noqa


def convert_to_west_state_dict(osum_state_dict):
    west_state_dict = {}
    for name in osum_state_dict.keys():
        if name.startswith("encoder."):
            new_name = name.replace("encoder.", "encoder.encoder.")
            west_state_dict[new_name] = osum_state_dict[name]
        elif name.startswith("speech_transformer."):
            new_name = name.replace("speech_transformer.",
                                    "projector.speech_transformer.")
            west_state_dict[new_name] = osum_state_dict[name]
        elif name.startswith("llama_model."):
            new_name = name.replace("llama_model.", "llm.")
            west_state_dict[new_name] = osum_state_dict[name]
        elif name.startswith("down_sample_2."):
            new_name = name.replace("down_sample_2.", "projector.")
            west_state_dict[new_name] = osum_state_dict[name]
        elif name.startswith("speech_llama_proj."):
            new_name = name.replace("speech_llama_proj.",
                                    "projector.speech_llama_proj.")
            west_state_dict[new_name] = osum_state_dict[name]
        elif name.startswith('speech_token_emded.'):
            west_state_dict[name] = osum_state_dict[name]

    return west_state_dict


def get_configs(llm_model_dir, wenet_model_dir, prompt_config_path):
    configs = {
        "llm_model_name_or_path": llm_model_dir,
        "lora_config": {
            "inference_mode": False,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "r": 8,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj"
            ],
            "task_type": "CAUSAL_LM"
        },
        "model_type": "osum",
        "no_init_llm": False,
        "projector_type": "transformer",
        "transformers_version": "4.52.3",
        "wenet_model_name_or_path": wenet_model_dir,
        "prompt_conf_path": prompt_config_path,
    }
    return configs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--osum_model_path", type=str, required=True)
    parser.add_argument("--llm_model_dir", type=str, required=True)
    parser.add_argument("--wenet_model_dir", type=str, required=True)
    parser.add_argument("--prompt_config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    checkpoint = torch.load(args.osum_model_path,
                            map_location="cpu",
                            weights_only=False)
    os.makedirs(args.output_dir)
    state_dict = convert_to_west_state_dict(checkpoint)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        configs = get_configs(args.llm_model_dir,
                              args.wenet_model_dir,
                              args.prompt_config_path)
        print(configs)
        json.dump(configs, f, indent=4)

    config = AutoConfig.from_pretrained(f'{args.output_dir}/config.json')
    model = AutoModel.from_config(config)
    tokenizer = model.init_tokenizer()
    print("Loading osum weights...")
    model.load_state_dict(state_dict, strict=False)
    print("Saving west model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
