# Copyright (c) 2025 Wenet Community. authors: Mddct(Dinghao Zhou)
#                                              Chengdong Liang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import json
import os

import torch
from transformers import AutoConfig, AutoModel

import west  # for init touchasu model.  # noqa


def convert_to_west_state_dict(firered_state_dict):
    wenet_state_dict = {}
    unused = []
    print(
        "===================== start CKPT Conversion ========================="
    )
    for name in firered_state_dict.keys():
        if 'llm.base_model' in name:
            wenet_state_dict[name] = firered_state_dict[name]
            continue
        original_name = copy.deepcopy(name)
        if 'input_preprocessor' in original_name:
            name = name.replace("input_preprocessor", "embed")
            name = name.replace('encoder.embed.out', 'encoder.embed.out.0')

        name = name.replace("decoder.token_embedding", "decoder.embed.0")
        name = name.replace("encoder.layer_stack", "encoder.encoders")
        name = name.replace("decoder.layer_stack", "decoder.decoders")
        # decoder attn
        name = name.replace(".cross_attn.w_qs", ".src_attn.linear_q")
        name = name.replace(".cross_attn.w_ks", ".src_attn.linear_k")
        name = name.replace(".cross_attn.w_vs", ".src_attn.linear_v")
        name = name.replace(".cross_attn.fc", ".src_attn.linear_out")
        name = name.replace(".self_attn.w_qs", ".self_attn.linear_q")
        name = name.replace(".self_attn.w_ks", ".self_attn.linear_k")
        name = name.replace(".self_attn.w_vs", ".self_attn.linear_v")
        name = name.replace(".self_attn.fc", ".self_attn.linear_out")
        # encoder attn
        name = name.replace(".mhsa.w_qs", ".self_attn.linear_q")
        name = name.replace(".mhsa.w_ks", ".self_attn.linear_k")
        name = name.replace(".mhsa.w_vs", ".self_attn.linear_v")
        name = name.replace(".mhsa.fc", ".self_attn.linear_out")
        name = name.replace(".mhsa.pos_bias_u", ".self_attn.pos_bias_u")
        name = name.replace(".mhsa.pos_bias_v", ".self_attn.pos_bias_v")
        name = name.replace(".mhsa.linear_pos", ".self_attn.linear_pos")

        # decoder mlp
        name = name.replace(".mlp.", ".feed_forward.")
        # encodr mlp
        name = name.replace(".ffn1.net.1", ".feed_forward_macaron.w_1")
        name = name.replace(".ffn1.net.4", ".feed_forward_macaron.w_2")
        name = name.replace(".ffn2.net.1", ".feed_forward.w_1")
        name = name.replace(".ffn2.net.4", ".feed_forward.w_2")

        # decoder pre norm
        name = name.replace(".self_attn_norm.", ".norm1.")
        name = name.replace(".cross_attn_norm.", ".norm2.")
        name = name.replace(".mlp_norm.", ".norm3.")
        # encoder pre norm
        name = name.replace(".ffn1.net.0.", ".norm_ff_macaron.")
        name = name.replace(".mhsa.layer_norm_q.", ".self_attn.layer_norm_q.")
        name = name.replace(".mhsa.layer_norm_k.", ".self_attn.layer_norm_k.")
        name = name.replace(".mhsa.layer_norm_v.", ".self_attn.layer_norm_v.")
        name = name.replace(".conv.pre_layer_norm.", ".norm_conv.")
        name = name.replace(".ffn2.net.0", ".norm_ff")
        name = name.replace(".layer_norm.", ".norm_final.")
        name = name.replace(".layer_norm.", ".norm_final.")

        # encoder conv
        if 'embed' not in name:
            name = name.replace(".conv.", ".conv_module.")
            name = name.replace(".batch_norm.", ".norm.")

        if "decoder" in name:
            name = name.replace("cross_attn_ln", "norm2")
            name = name.replace("mlp_ln", "norm3")
        else:
            name = name.replace("mlp_ln", "norm2")

        if original_name == "decoder.tgt_word_emb.weight":
            name = "decoder.embed.0.weight"
        if original_name == "decoder.tgt_word_prj.weight":
            name = "decoder.output_layer.weight"
        if 'decoder.layer_norm_out.' in original_name:
            name = name.replace('decoder.layer_norm_out', 'decoder.after_norm')

        if 'encoder_projector' in original_name:
            name = name.replace("encoder_projector.", "projector.")

        if 'encoder.' in name and 'positional_encoding' not in name:
            name = name.replace("encoder.", "encoder.encoder.")

        print("name  {} ==> {}".format(original_name, name))
        print("type  {} ==> torch.float32".format(
            firered_state_dict[original_name].dtype))
        print("shape {}\n".format(firered_state_dict[original_name].shape))
        if (original_name == name):
            unused.append(name)
        else:
            wenet_state_dict[name] = firered_state_dict[original_name].float()
    for name in unused:
        print("NOTE!!! drop {}".format(name))
    print(
        "DONE\n===================== End CKPT Conversion ====================\n"
    )
    return wenet_state_dict


def get_configs(llm_model_dir, wenet_model_dir):
    configs = {
        "encoder_ds_rate": 4,
        "encoder_projector_ds_rate": 2,
        "llm_model_name_or_path": llm_model_dir,
        "lora_config": {
            "inference_mode": False,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "r": 64,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj"
            ],
            "task_type": "CAUSAL_LM"
        },
        "model_type": "touch_asu",
        "projector_hidden_size": 3584,
        "projector_type": "linear",
        "transformers_version": "4.52.3",
        "wenet_model_name_or_path": wenet_model_dir,
    }
    return configs


def get_args():
    parser = argparse.ArgumentParser(description='load and parse whisper')
    # yapf: disable
    parser.add_argument(
        '--firered_model_dir',
        required=True,
        help='https://huggingface.co/FireRedTeam/FireRedASR-AED-L/tree/main'
    )
    # yapf: enable
    parser.add_argument('--output_dir',
                        default='./fireredasr-llm',
                        help='output file in west\'s style')
    parser.add_argument('--llm_model_dir',
                        default='Qwen/Qwen2-7B-Instruct')
    parser.add_argument('--wenet_model_dir',
                        default='firered')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    checkpoint = torch.load(os.path.join(args.firered_model_dir,
                                         'model.pth.tar'),
                            map_location="cpu",
                            weights_only=False)

    os.makedirs(args.output_dir)

    state_dict = convert_to_west_state_dict(
        checkpoint["model_state_dict"]
    )

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        configs = get_configs(args.llm_model_dir, args.wenet_model_dir)
        print(configs)
        json.dump(configs, f, indent=4)

    config = AutoConfig.from_pretrained(f'{args.output_dir}/config.json')
    print(f"Config: {config}")
    model = AutoModel.from_config(config)
    print(f"Model: {model}")
    tokenizer = model.init_tokenizer()
    print("Loading fireredasr-llm weights")
    model.load_state_dict(state_dict, strict=False)
    print("Saving fireredasr-llm weights to {}...".format(args.output_dir))
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
