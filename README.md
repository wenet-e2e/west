# WEST

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)
[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat&logo=wechat&logoColor=white)](#discussion--communication)

**We** **S**peech **T**ookit, LLM based Speech Toolkit for Speech Understanding,
Generation, and Interaction.

## Highlights

* **Fully LLM-based**: Standing on the shoulders of giants by reusing mature
  architectures, ecosystems (e.g., Hugging Face), and methods (e.g.,
  sequence packing) from large models.
* **Full-stack**: Supports tasks such as recognition, synthesis, understanding,
  dialogue, and multimodal capabilities, with extensibility to incorporate
  open-source models.

* **Simple and Stupid**: A simple and stupid speech toolkit that
  everyone can Touch.

## Install

``` bash
conda create -n west python=3.10
conda activate west
pip install -r requirements.txt
```

## Supported Tasks and Models

| Task                   | Model               | Recipe                                                                  |
|------------------------|---------------------|-------------------------------------------------------------------------|
| Speech Recognition     | TouchASU(Built-in)  | [aishell](examples/aishell/asr)                                         |
| Speech Synthesis       | TouchTTS(Built-in)  | [libritts](examples/libritts/tts)                                       |
| Speech QA              | TouchASU(Built-in)  | [belle_1.4M_qa](examples/belle_1.4M_qa)                                 |
| Speech Interaction     | TouchChat(Built-in) |                                                                         |
| MutliModal Interaction | TouchOmni(Built-in) |                                                                         |


## Citation

```

```

## Discussion & Communication

We created a WeChat group for better discussion and quicker response.
Please scan the personal QR code on the left, who is responsible for inviting you to the chat group.
You can also scan the QR code on the right to follow our official account of WeNet Community.

| <img src="https://raw.githubusercontent.com/robin1001/qr/master/chengdong.jpg" width="250px"> | <img src="https://raw.githubusercontent.com/robin1001/qr/master/wenet.jpeg" width="250px"> |
| ---- | ---- |
