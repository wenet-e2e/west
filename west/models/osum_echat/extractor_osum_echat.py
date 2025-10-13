# Copyright (c) 2025 Xuelong Geng(xlgeng@mail.nwpu.edu.cn)

from west.dataset.extractor import Extractor


class ExtractorOSUMEChat(Extractor):

    def __init__(self, tokenizer, model_config, inference=False):
        super().__init__(tokenizer, model_config, inference)
        # TODO(Xuelong Geng): Complete the design of extractor

    def extract(self, item):
        """
        TODO(Xuelong Geng): Complete the function
        """
