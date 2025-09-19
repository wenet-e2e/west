## Data Format

Inspired by the training paradigm of text LLM, the training of large speech models typically consists of two stages: pre-training and fine-tuning.
The pre-training stage usually utilizes large-scale datasets, while the fine-tuning stage employs smaller-scale datasets.
Typically, massive foundational speech data may contain speech and optional text annotations.
If the foundational data has corresponding text labels, some tasks use speech recognition and synthesis for pre-training, such as QWen2.5-Omni, Freeze-Omni, OSUM-eChat, etc.
Other tasks use self-supervised learning with speech tokens for pre-training, such as Moshi, Step-Audio, etc.
In the fine-tuning stage, a small amount of high-quality annotated data with more information and various task types is usually used.


To support the aforementioned two training stages and consider data storage and reading efficiency, we have designed the following two data formats:


* **Pre-training Data Format**: This data contains only speech and optional text labels, recorded in jsonl format.
    Each json includes two fields, `wav` and `txt`, corresponding to the speech file path and text label, respectively.
    Pre-training typically requires more than a million hours of data, containing over a billion speech files.
    To efficiently store and read massive foundational data, inspired by WeNet, we also support packaging multiple foundational data entries using the tar format.
    During training, only the list paths of all tar packages need to be provided.
    WEST will download and extract the tar packages on-the-fly and read the speech and text data within them.
    Data compression and packaging not only greatly improve data storage and reading efficiency but also eliminate the need to record the addresses of all audio files,
    thereby saving memory during training.

* **Fine-tuning Data Format**: We use the role-content format commonly used in the large model field.
    This format conveniently supports multi-turn interactions between users and systems and can be flexibly extended to modalities,
    which is crucial for tasks such as speech understanding, dialogue, and multi-turn interactions.
    We record all fine-tuning data in jsonl format.
    The *content* field can be either a text, an audio or mixed content.

Here shows some examples:

**Example: jsonl for Pre-training**
```
{"wav": "path/to/your/audio1.wav", "txt": "your text1 here"}
{"wav": "path/to/your/audio2.wav", "txt": "your text2 here"}
...
```

**Example: tar list for Pre-training**
```
path/to/your/data1.tar
path/to/your/data2.tar
...
# Note: each tar file contains multiple wavs and txts.
```

**Example of role-content based data for Fine-tuning**

``` json
{
   "messages":[
      {
         "role":"user",
         "content": "your text1 here"
      },
      {
         "role":"assistant",
         "content":{
            "type":"audio",
            "audio":"path/to/your/audio2.wav",
            "text":"your text2 here"
         }
      },
    {
         "role":"user",
         "content":{
            "type":"audio",
            "audio":"path/to/your/audio3.wav",
            "text":"your text3 here"
         }
      },
      {
         "role":"assistant",
         "content": "your text4 here"
      }
   ]
}

# Note: Here we just unwrap one json line for better readability.
```
