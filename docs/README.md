# Docs

## Install

``` bash
conda create -n west python=3.10
conda activate west
pip install -r requirements.txt
```

## Tutorial

Please checkout the tutorial in for each task in the table listed below.

| Task                   | Model               | Recipe                                                                  |
|------------------------|---------------------|-------------------------------------------------------------------------|
| Speech Recognition     | TouchASU(Built-in)  | [aishell](examples/aishell/asr)                                         |
| Speech Synthesis       | TouchTTS(Built-in)  | [libritts](examples/libritts/tts)                                       |
| Speech QA              | TouchASU(Built-in)  | [belle_1.4M_qa](examples/belle_1.4M_qa)                                 |
| Speech Interaction     | TouchChat(Built-in) |                                                                         |
| MutliModal Interaction | TouchOmni(Built-in) |                                                                         |


## Dive to WEST

* Data & Data Pipeline Design
  * [Data Format & Prepare](./data_format.md)
  * [Data pack(sequence pack)](./data_pack.md)
  * [Data Extractor](./data_extractor.md)
* Model Design
  * TODO
