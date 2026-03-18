# Copyright 2025 Hao Yin(1049755192@qq.com)

# whisper asr for compute wer.

import json
import sys

from tn_utils import get_tn_model, normalize_text
from tqdm import tqdm


def get_asr_model(language):
    if language == "en":
        import whisper
        return whisper.load_model("large-v3-turbo")
    elif language == "zh":
        from funasr import AutoModel
        return AutoModel(
            model="paraformer-zh",
            disable_update=True,
        )
    else:
        raise ValueError("Invalid language: {}".format(language))


def transcribe(audio_path, language, asr_model):
    if language == "en":
        result = asr_model.transcribe(audio_path, language=language)
        return result["text"]
    elif language == "zh":
        result = asr_model.generate(input=audio_path, batch_size_s=300)
        text = result[0]["text"]
        import zhconv
        return zhconv.convert(text, "zh-cn")
    else:
        raise ValueError("Invalid language: {}".format(language))


def main():
    wav_scp = sys.argv[1]
    syn_text = sys.argv[2]
    language = sys.argv[3]

    tn_model = get_tn_model(language)
    asr_model = get_asr_model(language)

    with open(wav_scp, "r") as f:
        with open(syn_text, "w") as out_f:
            for line in tqdm(f, desc="ASR transcribing..."):
                key, wav_path = line.strip().split("\t")
                text = transcribe(wav_path, language, asr_model).strip()
                text = normalize_text(text, language, tn_model)

                # save as JSONL format
                json_obj = {"key": key, "txt": text}
                out_f.write(json.dumps(json_obj) + "\n")


if __name__ == "__main__":
    main()
