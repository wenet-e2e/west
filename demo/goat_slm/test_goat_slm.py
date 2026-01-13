# -*- coding: utf-8 -*-
"""
@Author : songyd, chenhj
@File   : test_goat_slm.py
"""
import argparse
import json
import os
import re
import sys
import uuid

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import (AutoTokenizer, GenerationConfig,
                          WhisperFeatureExtractor)

sys.path.insert(0, '../../../west')
sys.path.insert(0, '../../west/models/goat_slm/CosyVoice')
sys.path.insert(0,
                '../../west/models/goat_slm/CosyVoice/third_party/Matcha-TTS')

from west.models.goat_slm import CosyVoice, GOATSLMModel  # noqa: E402


def load_wav(wav, target_sr, device=None):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert device is not None, 'device is None!!!'
        speech = speech.to(device=device)
        # assert sample_rate > target_sr, \
        #     'wav sample rate {} must be greater than {}'.format(
        #         sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr).cuda()(speech)
    return speech[:, :int(target_sr * 30)]


def process_dataset(batch, tokenizer, type='qwen', enable_thinking=False):
    # text branch
    text_llm_system_prompt = (
        "你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手，"
        "隶属于中国电信集团，基于中国电信星辰语义大模型，"
        "目标是与用户建立亲密和自然的对话关系，为用户提供温馨、贴心的聊天体验。"
        "请准确识别用户的情绪，如开心、难过、生气等，根据用户的情绪做出相应的回应，"
        "在用户感到难过时提供安慰，在用户开心时分享喜悦。"
    )
    # speech branch
    speech_llm_system_prompt = "You are a helpful assistant."
    if 'qwen' in type:
        text_llm_prefix = (
            f"<|im_start|>system\n{text_llm_system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        speech_llm_prefix = (
            f"<|im_start|>system\n{speech_llm_system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        if not enable_thinking:
            suffix += '<think>\n\n</think>\n\n'
        response = (
            batch["text_a"] + "<|im_end|>" if 'text_a' in batch
            else "文件中没有提供合适的回复内容。<|im_end|>"
        )
    else:
        text_llm_prefix = f"<_system>{text_llm_system_prompt}\n<_user>"
        speech_llm_prefix = f"<_system>{speech_llm_system_prompt}\n<_user>"
        suffix = "<_bot>"
        # if enable_thinking == False:
        #     suffix += '<think>\n\n</think>\n'
        response = (
            batch["text_a"] + "<_end>" if 'text_a' in batch
            else "文件中没有提供合适的回复内容。<_end>"
        )

    text_llm_input_ids = tokenizer.encode(text_llm_prefix)
    text_llm_attention_mask = [1] * len(text_llm_input_ids)
    labels = [-100] * len(text_llm_input_ids)

    speech_llm_input_ids = tokenizer.encode(speech_llm_prefix)
    speech_llm_attention_mask = [1] * len(speech_llm_input_ids)

    suffix_input_ids, suffix_attention_mask, suffix_labels = [], [], []
    new_input_ids = tokenizer.encode(suffix)
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    suffix_labels += [-100] * len(new_input_ids)

    answer_input_ids = tokenizer.encode(response)

    if "text_q" not in batch or batch["text_q"] == "":
        query_ids = [tokenizer.pad_token_id]
        query_attention_mask = [0]
        query_labels = [-100]
    else:
        query = batch["text_q"]
        new_input_ids = tokenizer.encode(query)
        query_ids = new_input_ids
        query_attention_mask = [1] * len(new_input_ids)
        query_labels = [-100] * len(new_input_ids)

    batch["input_ids"] = text_llm_input_ids
    batch["attention_mask"] = text_llm_attention_mask
    batch["labels"] = labels
    batch["speech_llm_input_ids"] = speech_llm_input_ids
    batch["speech_llm_attention_mask"] = speech_llm_attention_mask
    batch["suffix_input_ids"] = suffix_input_ids
    batch["suffix_attention_mask"] = suffix_attention_mask
    batch["suffix_labels"] = suffix_labels
    batch["answer_input_ids"] = answer_input_ids
    batch["query_ids"] = query_ids
    batch["query_attention_mask"] = query_attention_mask
    batch["query_labels"] = query_labels
    return batch


def init_cosyvoice(model_path):
    cosyvoice = CosyVoice(model_path, load_jit=False)
    return cosyvoice


reference_audio = './data/goat_slm_reference_speech.wav'
reference_speech_16k = load_wav(reference_audio, 16000, 'cuda')
reference_speech_22050 = load_wav(reference_audio, 22050, 'cuda')


def unit_to_wav(units, cosyvoice):
    flow_prompt_speech_token, reference_speech_token_len = (
        cosyvoice.frontend._extract_speech_token(reference_speech_16k)
    )  # [bs, t2]
    prompt_speech_feat, reference_speech_feat_len = (
        cosyvoice.frontend._extract_speech_feat(reference_speech_22050)
    )  # [bs, t1, 80]
    embedding = cosyvoice.frontend._extract_spk_embedding(reference_speech_16k)
    tts_speech = []
    for unit in units:
        tts_speech_token = torch.from_numpy(np.asarray(unit)).unsqueeze(0)
        this_uuid = str(uuid.uuid1())
        with cosyvoice.model.lock:
            cosyvoice.model.tts_speech_token_dict[
                this_uuid], cosyvoice.model.llm_end_dict[this_uuid] = [], False
            cosyvoice.model.mel_overlap_dict[
                this_uuid], cosyvoice.model.hift_cache_dict[
                    this_uuid] = None, None
        this_tts_speech = cosyvoice.model.token2wav(
            token=tts_speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=embedding,
            uuid=this_uuid,
            finalize=True,
            speed=1.0)
        tts_speech.append(this_tts_speech.cpu())
    tts_speech = torch.cat(tts_speech, dim=-1)
    return tts_speech


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        default=None,
                        help="Path to the input file",
                        required=True)
    parser.add_argument("--enable_thinking",
                        action="store_true",
                        help="Enable thinking mode")
    parser.add_argument("--output_file",
                        type=str,
                        default=None,
                        help="Path to the output file",
                        required=True)
    parser.add_argument("--slm_model_path",
                        type=str,
                        default=None,
                        help="Path to the slm model",
                        required=True)
    # args for generation
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=2048,
                        help="max new tokens for generation")
    parser.add_argument("--min_new_tokens",
                        type=int,
                        default=3,
                        help="min new tokens for generation")
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="whether do sample. "
             "For ST task, we will use greedy search to ensure stable output"
    )
    parser.add_argument("--temperature",
                        type=float,
                        default=0.9,
                        help="temperature for generation")
    parser.add_argument("--top_p",
                        type=float,
                        default=0.75,
                        help="top_p for generation")
    parser.add_argument("--top_k",
                        type=int,
                        default=20,
                        help="top_k for generation")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # ************************* custom params *************************
    enable_thinking = args.enable_thinking
    print(f"enable_thinking: {enable_thinking}")
    max_speech_unit_length = 50 * 50
    is_speech_generate_run = True
    is_teacher_forcing = False
    # *****************************************************************

    Y_model, loading_info_speech = GOATSLMModel.from_pretrained(
        args.slm_model_path,
        torch_dtype=torch.bfloat16,
        _fast_init=True,
        output_loading_info=True)
    Y_model = Y_model.cuda()
    device = Y_model.device
    Y_model.eval()

    assert (Y_model.config.model_type not in (
        'GOAT-SLM1-7B', 'GOAT-SLM2-1.8B')) or enable_thinking, \
        ("GOAT-SLM1-7B或者GOAT-SLM2-1.8B，enable_thinking必须为True。"
         "注意：GOAT-SLM1-7B不支持think，enable_thinking必须为True便于代码层面统一; "
         "GOAT-SLM2-1.8B只支持think模式。")

    fbank_extractor = WhisperFeatureExtractor.from_pretrained(
        f'{args.slm_model_path}/whisper-small')
    tokenizer = AutoTokenizer.from_pretrained(args.slm_model_path,
                                              trust_remote_code=True)

    # 初始化cosyvoice
    cosyvoice = init_cosyvoice(f"{args.slm_model_path}/CosyVoice-300M-SFT")

    generation_config = {
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,
        "eos_token_id": [
            tokenizer.eos_token_id,
        ],
        "repetition_penalty": 1.05,  # 1.5,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "output_hidden_states": True,
        "return_dict_in_generate": True
    }
    if 'qwen' in Y_model.llm_config.model_type.lower():
        skip_special_tokens = True
        think_template = '</think>\n\n'
    else:  # telechat
        skip_special_tokens = False
        think_template = '</think>\n'
    generation_config = GenerationConfig.from_dict(generation_config)
    print(generation_config)

    dataset = []
    with open(args.input_file, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            dataset.append(item)

    REGEX_HEAD = re.compile(r".*[Aa]sistant:")

    wav_dir = f'{args.output_file}/wavs'
    os.makedirs(wav_dir, exist_ok=True)

    with open(f"{args.output_file}/output.json", "w") as fout:
        for index, data_ in enumerate(dataset):
            data = process_dataset(data_,
                                   tokenizer,
                                   type=Y_model.llm_config.model_type.lower(),
                                   enable_thinking=enable_thinking)
            audio = data.get("speech_q", None)
            speech_values, speech_attention_mask = None, None
            if audio is not None:
                speech = load_wav(audio,
                                  target_sr=fbank_extractor.sampling_rate)
                speech_inputs = fbank_extractor(
                    speech.squeeze(0),
                    sampling_rate=fbank_extractor.sampling_rate,
                    return_attention_mask=True,
                    return_tensors='pt'  # "pt", 'np'
                )
                speech_values = speech_inputs.input_features.to(
                    device=device, dtype=torch.bfloat16)
                speech_attention_mask = speech_inputs.attention_mask.to(device)

            text_tokens, speech_units = Y_model.generate_custom(
                input_ids=torch.tensor(data['input_ids'],
                                       dtype=torch.int,
                                       device=device).unsqueeze(0),
                query_ids=torch.tensor(data['query_ids'],
                                       dtype=torch.int,
                                       device=device).unsqueeze(0),
                suffix_input_ids=torch.tensor(data['suffix_input_ids'],
                                              dtype=torch.int,
                                              device=device).unsqueeze(0),
                speech_values=speech_values,
                speech_attention_mask=speech_attention_mask,
                generation_config=generation_config,
                streamer=None,  # streamer,
                text_label=torch.tensor(data['answer_input_ids'],
                                        dtype=torch.int,
                                        device=device),
                is_teacher_forcing=is_teacher_forcing,
                speech_llm_input_ids=torch.tensor(data['speech_llm_input_ids'],
                                                  dtype=torch.int,
                                                  device=device).unsqueeze(0),
                max_speech_unit_length=max_speech_unit_length,
                is_speech_generate_run=is_speech_generate_run,
                enable_thinking=enable_thinking)

            response = tokenizer.decode(text_tokens.squeeze(0),
                                        skip_special_tokens=skip_special_tokens)
            response = re.sub(r'!{2,}', '', REGEX_HEAD.sub("", response))
            print(f"index: {index} |||  response text:", response)

            if think_template in response:
                think = response.split(think_template)[0].replace('<think>', '')
                response = response.split(think_template)[-1]
            else:
                think = ''

            key = data['key']
            if speech_units.size(-1) != 0:
                output_units = speech_units[0].cpu().tolist()
                if ('粤语' in think or '河南' in think or '东北' in think
                        or '四川' in think or '上海' in think):
                    output_units = output_units[17:]
                wav = unit_to_wav([output_units], cosyvoice)
                torchaudio.save(f'{wav_dir}/{key}.wav', wav, 22050)

            json_string = json.dumps(
                {
                    "key": f"{key}",
                    'pred_audio': f"{wav_dir}/{key}.wav",
                    "response": response,
                    'think': think,
                },
                ensure_ascii=False)
            fout.write(json_string + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
'''
--input_file ./data/goat_slm_test.json
--output_file xxx/result/
--slm_model_path xxx/GOAT-SLM2-8B
'''
