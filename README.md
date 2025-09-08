# WeST

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
| Speech Recognition     | TouchASU(Built-in)  | [aishell](examples/aishell/asr),[librispeech](examples/librispeech/asr) |
| Speech Synthesis       | TouchTTS(Built-in)  | [libritts](examples/libritts/tts)                                       |
| Speech Understanding   | TouchASU(Built-in)  |                                                                         |
| Speech Interaction     | TouchChat(Built-in) |                                                                         |
| MutliModal Interaction | TouchOmni(Built-in) |                                                                         |
