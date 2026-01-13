## 👉🏻 GOAT-SLM 👈🏻
https://tele-ai.github.io/GOAT-SLM.github.io/case/demo.mp4

## Highlight🔥
GOAT-SLM has been upgraded to Version 2 (GOAT-SLM2). This repository provides code inference support for both GOAT-SLM1 and GOAT-SLM2.


## Install

### Clone and install
- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
    ``` sh
    conda install pytorch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -c pytorch

    git clone git@github.com:Tele-AI/GOAT-SLM.git
    cd GOAT-SLM
    pip install -r requirements.txt
    ```

  If flash-attn fails to install, you can execute the following operations separately.
  ``` python
  pip install flash-attn==2.7.4.post1 --no-build-isolation
   ```

### Model download
| Models         | 🤗 Hugging Face |
|----------------|-------|
| GOAT-SLM1-7B   | [Tele-AI/GOAT-SLM1-7B](https://huggingface.co/Tele-AI/GOAT-SLM1-7B) |
| GOAT-SLM2-1.8B | [Tele-AI/GOAT-SLM2-1.8B](https://huggingface.co/Tele-AI/GOAT-SLM1-7B) |
| GOAT-SLM2-8B   | [Tele-AI/GOAT-SLM2-8B](https://huggingface.co/Tele-AI/GOAT-SLM1-7B) |

Note: The model size indicated in the model name is determined by the parameter scale of its underlying base large language model (Base LLM). For example, GOAT-SLM2-8B uses Qwen3-8B as its base LLM; therefore, its model size is labeled as 8B.

``` python
from huggingface_hub import snapshot_download
snapshot_download('Tele-AI/GOAT-SLM1-7B', local_dir='pretrained_models/GOAT-SLM')
snapshot_download('Tele-AI/GOAT-SLM2-1.8B', local_dir='pretrained_models/GOAT-SLM')
snapshot_download('Tele-AI/GOAT-SLM2-8B', local_dir='pretrained_models/GOAT-SLM')

# Dependent external model
snapshot_download('FunAudioLLM/CosyVoice-300M-SFT', local_dir='pretrained_models/GOAT-SLM/CosyVoice-300M-SFT')
snapshot_download('openai/whisper-small', local_dir='pretrained_models/GOAT-SLM/whisper-small')
```

### Quick start

**Interactive demo**
```sh
bash run_demo_goat_slm.sh
```
This launches a local Gradio interface where you can try GOAT-SLM interactively.

![Demo UI](assets/demo_ui.png)
``` python
root_path='pretrained_models/GOAT-SLM'
python3 demo_goat_slm.py \
    --slm_model_path $root_path \
    --max_new_tokens 256 \
    --port 8085 \
    --max_turn_num 6 \
    --is_speech_generate_run
```

**Local test**

- cli
``` python
python test_goat_slm.py --input_file ./data/goat_slm_test.json  \
--output_file ./result/ \
--slm_model_path pretrained_models/GOAT-SLM
```

## Acknowledge

1. We borrowed a lot of code from [blsp](https://github.com/cwang621/blsp).
2. We borrowed a lot of code from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice).

## Citations

``` bibtex
@article{chen2025goat,
  title={GOAT-SLM: A spoken language model with paralinguistic and speaker characteristic awareness},
  author={Chen, Hongjie and Li, Zehan and Song, Yaodong and Deng, Wenming and Yao, Yitong and Zhang, Yuxin and Lv, Hang and Zhu, Xuechao and Kang, Jian and Lian, Jie and others},
  journal={arXiv preprint arXiv:2507.18119},
  year={2025}
}

@article{li2025televal,
  title={TELEVAL: A dynamic benchmark designed for spoken language models in chinese interactive scenarios},
  author={Li, Zehan and Chen, Hongjie and Zhang, Yuxin and Zhou, Jing and Wang, Xuening and Lv, Hang and Du, Mengjie and Song, Yaodong and Lian, Jie and Kang, Jian and others},
  journal={arXiv preprint arXiv:2507.18061},
  year={2025}
}

@article{wang2025boss,
  title={Boss: Beyond-semantic speech},
  author={Wang, Qing and Li, Zehan and Lv, Hang and Chen, Hongjie and Song, Yaodong and Kang, Jian and Lian, Jie and Li, Jie and Li, Yongxiang and He, Zhongjiang and others},
  journal={arXiv preprint arXiv:2507.17563},
  year={2025}
}

@article{song2025goat,
  title={GOAT-TTS: Expressive and Realistic Speech Generation via A Dual-Branch LLM},
  author={Song, Yaodong and Chen, Hongjie and Lian, Jie and Zhang, Yuxin and Xia, Guangmin and Li, Zehan and Zhao, Genliang and Kang, Jian and Li, Jie and Li, Yongxiang and others},
  journal={arXiv preprint arXiv:2504.12339},
  year={2025}
}
}
```
