# -*- coding: utf-8 -*-
"""
@Author : songyd, chenhj
@File   : chat_demo_goat_slm.py
"""
import argparse
import base64
import datetime
import io
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from gradio import processing_utils
from transformers import (AutoTokenizer, GenerationConfig,
                          WhisperFeatureExtractor)

sys.path.insert(0, '../../../west')
sys.path.insert(0, '../../west/models/goat_slm/CosyVoice')
sys.path.insert(0,
                '../../west/models/goat_slm/CosyVoice/third_party/Matcha-TTS')
from west.models.goat_slm import CosyVoice, GOATSLMModel  # noqa: E402

generation_config = GenerationConfig(max_new_tokens=512,
                                     min_new_tokens=10,
                                     do_sample=True,
                                     temperature=0.9,
                                     top_p=0.7,
                                     num_beams=1,
                                     num_return_sequences=1,
                                     return_dict_in_generate=True,
                                     output_hidden_states=True,
                                     repetition_penalty=1.05,
                                     top_k=20)
dtype = torch.bfloat16
text_llm_system_prompt = (
    "你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手，"
    "隶属于中国电信集团，基于中国电信星辰语义大模型，"
    "目标是与用户建立亲密和自然的对话关系，为用户提供温馨、贴心的聊天体验。"
    "请准确识别用户的情绪，如开心、难过、生气等，根据用户的情绪做出相应的回应，"
    "在用户感到难过时提供安慰，在用户开心时分享喜悦。"
)


def load_wav(wav, target_sr, device=None):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert device is not None, 'device is None!!!'
        speech = speech.to(device=device)
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr).cuda()(speech)
    return speech[:, :int(target_sr * 30)].cpu()


pre_save_file = "./result/chat/wavs"
os.makedirs(pre_save_file, exist_ok=True)


def save_audio(audio_data,
               sample_rate,
               to_dir='./result/chat/wavs',
               to_name=None):
    # 获取当前日期和时间
    now = datetime.datetime.now()
    date_folder = f'{to_dir}/{now.strftime("%Y%m%d")}'
    if to_name is not None:
        time_filename = now.strftime("%H%M%S") + f"_{to_name}_q.wav"
    else:
        time_filename = now.strftime("%H%M%S") + "_q.wav"
    # 创建日期文件夹
    if not os.path.exists(date_folder):
        os.makedirs(date_folder)
    # 构建完整的文件路径
    file_path = os.path.join(date_folder, time_filename)
    global pre_save_file
    pre_save_file = file_path
    # audio_data = torch.from_numpy(audio_data).unsqueeze(0)
    torchaudio.save(file_path, audio_data, sample_rate)
    print(f"音频已保存到: {file_path}")


def save_tmp_audio(audio, cache_dir):
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False,
                                     suffix=".wav") as temp_audio:
        temp_audio.write(audio)
    return temp_audio.name


class ChatHistory(object):

    def __init__(self, tokenizer, extractor):
        super().__init__()
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.history = []
        self.audio_file = []
        self.audio_to_history = True

        self.add_sys_prompt(text_llm_system_prompt)

    def add_sys_prompt(self, sys_prompt):
        message = [{"role": "system", "content": f"{sys_prompt}"}]
        input_ids = tokenizer.apply_chat_template(message,
                                                  tokenize=True,
                                                  add_generation_prompt=False,
                                                  return_tensors="pt").cuda()
        # input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
        self.history.append((input_ids, ))

    def add_text_history(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.cuda()
        self.history.append((input_ids, ))

    def add_audio(self, audio_file):
        self.audio_to_history = False
        self.audio_file.append(audio_file)

    def add_speech_history(self, speech):
        if self.audio_to_history:
            return
        self.audio_to_history = True
        # print(speech)
        speech_name = speech.split('/')[-1].split('.')[0]

        speech = load_wav(speech,
                          target_sr=self.extractor.sampling_rate,
                          device='cuda')
        save_audio(speech, self.extractor.sampling_rate, to_name=speech_name)
        speech_inputs = self.extractor(
            speech.squeeze(0),
            sampling_rate=self.extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt")
        speech_values = speech_inputs.input_features.to(dtype).cuda()
        speech_attention_mask = speech_inputs.attention_mask.cuda()
        self.history.append((speech_values, speech_attention_mask))


def parse_args():
    parser = argparse.ArgumentParser(description="Chat Demo")
    parser.add_argument("--enable_thinking",
                        action="store_true",
                        help="Enable thinking mode")
    parser.add_argument("--is_speech_generate_run",
                        action="store_true",
                        help="Enable thinking mode")
    parser.add_argument("--slm_model_path",
                        type=str,
                        default=None,
                        help="Root path to the goat_slm",
                        required=True)
    # args for generation
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=512,
                        help="max new tokens for generation")
    parser.add_argument("--min_new_tokens",
                        type=int,
                        default=10,
                        help="min new tokens for generation")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.9,
                        help="temperature for generation")
    parser.add_argument("--top_p",
                        type=float,
                        default=0.7,
                        help="top_p for generation")
    parser.add_argument("--port", type=int, default=8080, help="port")
    parser.add_argument("--max_turn_num",
                        type=int,
                        default=6,
                        help="max turn num")
    parser.add_argument("--cache-dir",
                        type=str,
                        default="/tmp/TeleAdapterSLM",
                        help="Cache directory.")
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================
print('Initializing Chat')
args = parse_args()
Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
# ************************* custom params *************************
enable_thinking = args.enable_thinking
max_speech_unit_length = 50 * 50
is_speech_generate_run = args.is_speech_generate_run
is_teacher_forcing = False
# ***********************************************************************************************************

tokenizer = AutoTokenizer.from_pretrained(args.slm_model_path,
                                          trust_remote_code=True)

print(f"load checkpoint from {args.slm_model_path}")
Y_model, loading_info_speech = GOATSLMModel.from_pretrained(
    args.slm_model_path,
    torch_dtype=torch.bfloat16,
    _fast_init=True,
    output_loading_info=True)
Y_model = Y_model.to(dtype).cuda()
Y_model.eval()

whisper_model_path = f'{args.slm_model_path}/whisper-small'
print(f"load whisper feature extractor from {whisper_model_path}")
extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_path)

generation_config.update(
    **{
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id  # 151645 #
    })
speech_llm_system_prompt = "You are a helpful assistant."
if Y_model.config.model_type == 'GOAT-SLM2-1.8B':
    speech_llm_prefix = f"<_system>{speech_llm_system_prompt}\n"
else:
    speech_llm_prefix = (
        f"<|im_start|>system\n{speech_llm_system_prompt}<|im_end|>\n"
    )
speech_llm_input_ids = tokenizer.encode(speech_llm_prefix)


# ************************* flow + vocoder *************************
def init_cosyvoice(model_path):
    cosyvoice_model = CosyVoice(model_path, load_jit=False)
    return cosyvoice_model


reference_audio = './data/goat_slm_reference_speech.wav'
reference_speech_16k = load_wav(reference_audio, 16000, 'cuda')
reference_speech_22050 = load_wav(reference_audio, 22050, 'cuda')
cosyvoice = init_cosyvoice(f"{args.slm_model_path}/CosyVoice-300M-SFT")
flow_prompt_speech_token, reference_speech_token_len = (
    cosyvoice.frontend._extract_speech_token(reference_speech_16k)
)  # [bs, t2]
prompt_speech_feat, reference_speech_feat_len = (
    cosyvoice.frontend._extract_speech_feat(reference_speech_22050)
)  # [bs, t1, 80]
embedding = cosyvoice.frontend._extract_spk_embedding(reference_speech_16k)


def unit_to_wav(units):
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


# *******************************************************************************************************
history = ChatHistory(tokenizer, extractor)
logging.info(history.history)
print('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chatbot, sys_prompt):
    history.history = []
    history.audio_file = []
    history.add_sys_prompt(sys_prompt)
    return chatbot


def save_text(file_path, text):
    with open(file_path, 'w') as f:
        f.writelines(f"{text}\n")


def gradio_answer(chatbot, enable_thinking, is_his_empty_think,
                  is_speech_generate_run, do_sample, max_turn_num,
                  max_new_tokens, sys_prompt):
    generation_config.update(
        **{
            'do_sample': do_sample,
            'max_new_tokens': max_new_tokens
            # "temperature": temperature,
        })
    # max_turn_num = args.max_turn_num
    cur_turn_num = 0
    print(
        '******************** print history start: ********************'
    )
    for h in history.history:
        if len(h) == 1:
            # text
            input_ids = h[0]
            text = tokenizer.decode(input_ids.cpu().tolist()[0])
            print(f"text:{text}", end='')
            if '<|im_start|>user' in text or '<_user>' in text:
                cur_turn_num += 1
        elif len(h) == 2:
            # speech
            speech_values, _ = h[0], h[1]
            print(f"speech: {speech_values.size()}", end='')
    print(
        '******************** print history end!! ********************'
    )

    text_tokens, speech_units = Y_model.chat(
        history=history.history,
        generation_config=generation_config,
        speech_llm_input_ids=torch.tensor(speech_llm_input_ids,
                                          dtype=torch.int,
                                          device="cuda").unsqueeze(0),
        max_speech_unit_length=max_speech_unit_length,
        is_speech_generate_run=is_speech_generate_run,
        enable_thinking=enable_thinking)
    response = tokenizer.decode(text_tokens[0], skip_special_tokens=False)
    print(f"response: {response}\n")
    # 清除speech history
    # del history.history[-2]
    if is_his_empty_think and '</think>\n' in response:
        # response_wo_think = '<think>\n\n</think>\n\n' + \
        #     response.split('</think>\n\n')[-1]
        response_wo_think = response.split('</think>\n')[-1]
        history.add_text_history(response_wo_think + "\n")
    else:
        history.add_text_history(response + "\n")

    if enable_thinking:
        response = response.replace("<", "&lt;").replace(">", "&gt;")
    response = f'Turn-{cur_turn_num}:\n{response}'
    if cur_turn_num >= max_turn_num:
        gradio_reset(chatbot, sys_prompt)
        response += '\n\n多轮对话已重置！！！'

    if speech_units.size(-1) != 0:
        output_units = speech_units[0].cpu().tolist()
        if enable_thinking and ('粤语' in response or '河南' in response
                                or '东北' in response or '四川' in response
                                or '上海' in response):
            output_units = output_units[17:]  # + output_units[]
        print(
            f"response text len: {len(text_tokens[0])}, "
            f"unit len: {len(output_units)//4}"
        )
        wav_full = unit_to_wav([output_units])

        global pre_save_file
        pre_save_file = pre_save_file.replace('_q.wav', '_a.wav')
        torchaudio.save(pre_save_file, wav_full, 22050)
        save_text(pre_save_file.replace('.wav', '.txt'), response)

        wav_full = wav_full[0].numpy()
        # else:
        #     wav_full = np.zeros((160000,), dtype=np.float32)

        # 写入内存 buffer
        buffer = io.BytesIO()
        sf.write(buffer, wav_full, 22050, format="WAV")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()

        reply_html = f"""
<div>
    <p>{response}</p>
    <audio controls>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
</div>
        """
    else:
        reply_html = response

    chatbot[-1][1] = reply_html
    # for character in response:
    #     chatbot[-1][1] += character
    # print(f"chatbot: {chatbot}")
    # return_value = (22050, wav_full)
    yield (chatbot)  # return_value


title = """<h1 align="center">GOAT-SLM</h1>"""
description = (
    """<h3>This is the demo of GOAT-SLM. """
    """Upload your audios and start chatting!</h3>"""
)


def add_text(chatbot, user_message, enable_thinking, model_type):
    chatbot = chatbot + [(user_message, None)]
    logging.info(f"user_message: {user_message}")
    logging.info(f"chatbot: {chatbot}")
    # chatbot = chatbot + [user_message]
    if model_type == 'GOAT-SLM2-1.8B':
        user_message = "<_user>" + user_message + "<_bot>"
    else:
        user_message = (
            "<|im_start|>user\n" + user_message +
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        if not enable_thinking:
            user_message += '<think>\n\n</think>\n\n'
    history.add_text_history(user_message)
    return chatbot, gr.update(value="", interactive=False)


def add_file(chatbot, gr_audio, enable_thinking, model_type):
    print(f'gr_audio.name: {gr_audio.name}')
    if model_type == 'GOAT-SLM2-1.8B':
        history.add_text_history("<_user>")
        user_message = "<_bot>"
    else:
        history.add_text_history(
            "<|im_start|>user\n")  # f"<|im_start|>user\n{transcription}"
        user_message = "<|im_end|>\n<|im_start|>assistant\n"
        if not enable_thinking:
            user_message += '<think>\n\n</think>\n\n'
    history.add_audio(gr_audio.name)
    history.add_speech_history(history.audio_file[-1])
    chatbot = chatbot + [((gr_audio.name, ), None)]
    history.add_text_history(user_message)
    return chatbot


def add_micophone_file(chatbot, gr_audio_mic, enable_thinking, model_type):
    if gr_audio_mic is not None:
        audio = processing_utils.audio_from_file(gr_audio_mic)
        # audio_ = processing_utils.convert_to_16_bit_wav(audio[1])
        processing_utils.audio_to_file(audio[0], audio[1],
                                       gr_audio_mic + '.wav')
        # os.rename(gr_audio_mic, gr_audio_mic + '.wav')
        gr_audio_mic_wav = gr_audio_mic + ".wav"

        print(f'gr_audio_mic_wav: {gr_audio_mic_wav}')

        if model_type == 'GOAT-SLM2-1.8B':
            history.add_text_history("<_user>")
            user_message = "<_bot>"
        else:
            history.add_text_history(
                "<|im_start|>user\n")  # # f"<|im_start|>user\n{transcription}"
            user_message = "<|im_end|>\n<|im_start|>assistant\n"
            if not enable_thinking:
                user_message += '<think>\n\n</think>\n\n'

        history.add_audio(gr_audio_mic_wav)
        history.add_speech_history(history.audio_file[-1])
        chatbot = chatbot + [((gr_audio_mic_wav, ), None)]
        history.add_text_history(user_message)
    return chatbot, gr.update(value=None, interactive=True)


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    chatbot = gr.Chatbot([],
                         elem_id="chatbot",
                         height=750,
                         avatar_images=(None,
                                        (os.path.join(os.path.dirname(__file__),
                                                      "data/goat.png"))))
    gr_model_type = gr.Textbox(value=Y_model.config.model_type, visible=False)
    with gr.Row():
        with gr.Column(scale=0.2, min_width=0):
            is_speech_generate_run = gr.Checkbox(
                label="speech generate run?", value=args.is_speech_generate_run)
            do_sample = gr.Checkbox(label="do sample?", value=True)
            enable_thinking = gr.Checkbox(
                label="enable thinking?",
                value=True,
                visible=Y_model.config.model_type == 'GOAT-SLM2-8B')
            is_his_empty_think = gr.Checkbox(
                label="history empty_think?",
                value=False,
                visible=Y_model.config.model_type == 'GOAT-SLM2-8B')
            max_turn_num = gr.Slider(
                minimum=1,
                maximum=20,
                value=args.max_turn_num,
                step=1,
                interactive=True,
                label="turn_num",
            )
            max_new_tokens = gr.Slider(
                minimum=128,
                maximum=2048,
                value=1024,
                step=10,
                interactive=True,
                label="max_text_new_tokens",
            )
        with gr.Column(scale=0.10, min_width=0):
            clear = gr.Button("re_pro_and_con")  # reset prompt and context
        with gr.Column(scale=0.65):
            with gr.Row():
                sys_prompt = gr.Textbox(label="System Prompt",
                                        value=f"{text_llm_system_prompt}",
                                        lines=2)
            with gr.Row():
                txt = gr.Textbox(show_label=False,
                                 placeholder="Enter text and press enter",
                                 container=False)
        with gr.Column(scale=0.08, min_width=0):
            btn = gr.UploadButton("📁", file_types=["video", "audio"])
        with gr.Column(scale=0.2, min_width=0):
            input_audio_mic = gr.Audio(
                label="🎤",
                type="filepath",
                sources="microphone",
                visible=True,
            )
        # with gr.Column(scale=0.5):
        #     audio_output_box = gr.Audio(label="Speech Output")
        # unit_output_box = gr.Textbox(label="Unit Output", type="text")

    txt_msg = txt.submit(add_text,
                         [chatbot, txt, enable_thinking, gr_model_type],
                         [chatbot, txt],
                         queue=False).then(gradio_answer, [
                             chatbot, enable_thinking, is_his_empty_think,
                             is_speech_generate_run, do_sample, max_turn_num,
                             max_new_tokens, sys_prompt
                         ], [chatbot])
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    file_msg = btn.upload(add_file,
                          [chatbot, btn, enable_thinking, gr_model_type],
                          [chatbot],
                          queue=False).then(gradio_answer, [
                              chatbot, enable_thinking, is_his_empty_think,
                              is_speech_generate_run, do_sample, max_turn_num,
                              max_new_tokens, sys_prompt
                          ], [chatbot])

    input_audio_mic.change(
        add_micophone_file,
        [chatbot, input_audio_mic, enable_thinking, gr_model_type],
        [chatbot, input_audio_mic],
        queue=True).then(gradio_answer, [
            chatbot, enable_thinking, is_his_empty_think,
            is_speech_generate_run, do_sample, max_turn_num, max_new_tokens,
            sys_prompt
        ], [chatbot])
    # clear.click(gradio_reset, [chatbot, sys_prompt],
    #             [chatbot, txt, input_audio_mic, btn], queue=False)
    clear.click(gradio_reset, [chatbot, sys_prompt], [chatbot], queue=False)

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=args.port,
    # share=True,
    # pwa=True,
)
