# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)

import torch


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def non_causal_mask(lengths):
    """
    Args:
        lengths: (B,), such as [3, 5, 2]
    Returns:
        attention_mask: (B, max_len, max_len), padding is False
    """
    batch_size = lengths.size(0)
    max_len = lengths.max().item()
    mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool)
    for k, length in enumerate(lengths):
        mask[k, :length, :length] = True
    return mask


# print(non_causal_mask(torch.tensor([2, 3, 4], dtype=torch.long)))
