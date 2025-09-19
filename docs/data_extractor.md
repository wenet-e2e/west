## Data Extactor

We designed Model Extractor for each model to extract the inputs required by that model.
Each Model Extractor inherits from the `Extractor` defined by WEST, as shown below:

``` python
class Extractor(ABC):

    model_type = 'model'

    # Batch/Pack fileds for dataset
    fields_batch_static = {}
    fields_batch_dynamic = {}
    fields_pack_offset = {}

    def __init__(self, tokenizer, model_config, inference=False):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.inference = inference

    @abstractmethod
    def extract(self, item):
        pass
```

* `fields_batch_static`: fileds which has static shape, such as text lengths, speech feature lengths.
* `fields_batch_dynamic`: fileds which has dynamic shape, which requires padding when batching, such as speech feature.
* `fields_pack_offset`: fileds we want to keep offset record when packing sequence, such as the speech token start index.

The `extract` function returns a dict object, and here `fileds` are keys in the returned dict.


Here is an example of `TouchTTS` Extactor.


``` python

class ExtractorTouchTTS(Extractor):
    model_type = 'touch_tts'
    fields_batch_static = {'audio_offsets', 'text_lengths'}
    fields_batch_dynamic = {'audio_features', 'input_ids', 'labels'}
    fields_pack_offset = {'audio_offsets'}

    def extract(self, item):
        import s3tokenizer
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        waveform, sample_rate = torchaudio.load(item['wav'])
        audio = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        audio = audio[0]  # get the first channel
        mel = s3tokenizer.log_mel_spectrogram(audio)
        mel = mel.transpose(0, 1)
        # There is 100 frames mel per second, and 25 tokens per second
        num_audio_token = math.ceil(mel.size(0) * 25 / 100.0 - 1e-9)
        if not self.inference:
            content = item['txt'] + '<|audio_bos|>'
            token_lengths = 0
        else:
            content = item['txt'] + item['syn_txt'] + '<|audio_bos|>'
            token_lengths = len(self.tokenizer.encode(item['syn_txt']))
        ids_text = [self.tokenizer.bos_token_id
                    ] + self.tokenizer.encode(content)
        tgt_text = [IGNORE_TOKEN_ID] * len(ids_text)
        ids_audio = [0] * num_audio_token
        if not self.inference:
            ids = ids_text + ids_audio + [self.tokenizer.eos_token_id]
            tgt = tgt_text + ids_audio + [self.tokenizer.eos_token_id]
        else:
            ids = ids_text + ids_audio
            tgt = tgt_text + ids_audio
        input_ids = torch.tensor(ids, dtype=torch.long)
        tgt_ids = torch.tensor(tgt, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': tgt_ids,
            'audio_features': mel,
            'audio_offsets': len(ids_text),
            'text_lengths': token_lengths
        }
```


The `ExtractorTouchTTS` returns a dict and in which:

* `input_ids`, `labels`, `audio_features` are in dyanmic shape.
* `audio_offsets`, `text_lengths` are in static shape.
* `audio_offsets` is the speech token start offset.
