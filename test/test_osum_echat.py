import torch


from transformers import (AutoConfig, AutoModel)

import west.models.osum_echat.patch4generate
if __name__ == '__main__':
    osum_config_path = ("/Users/xuelonggeng/Documents/code/west_xlgeng/examples/aishell/asr/conf/osum_echat.json")
    config_new = AutoConfig.from_pretrained(osum_config_path)
    print(config_new)
    osum_model = AutoModel.from_config(config_new)
    print(osum_model)
    fake_wav = torch.randn(1,121, 80)
    faek_wav_lens = torch.LongTensor([121])
    osum_output = osum_model.generate(audio_features=fake_wav, audio_features_lengths=faek_wav_lens)
    print(osum_output)