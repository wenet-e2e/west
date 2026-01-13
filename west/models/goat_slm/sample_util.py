# -*- coding: utf-8 -*-
"""
@Author : songyd, chenhj
@File   : sample_util.py
"""
from typing import Optional, Tuple

import torch


def multinomial_sample_one_no_sync_ori(probs_sort, index=None):
    q = torch.empty_like(probs_sort,
                         dtype=probs_sort.dtype,
                         device=probs_sort.device,
                         requires_grad=False).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1,
                        keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.0,
):
    # if previous_tokens is not None:
    #     previous_tokens = previous_tokens.squeeze()
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=1, index=previous_tokens)
        score = torch.where(score < 0, score * repetition_penalty,
                            score / repetition_penalty)
        logits = logits.scatter(dim=1, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits,
                                                             dim=-1),
                                 dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=0, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    index=None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(logits=logits,
                            previous_tokens=previous_tokens,
                            **sampling_kwargs)
    idx_next = multinomial_sample_one_no_sync_ori(probs, index=index)
    return idx_next, probs


# Repetition Aware Sampling in VALL-E 2
def ras_sampling(weighted_scores,
                 decoded_tokens,
                 top_p=0.8,
                 top_k=25,
                 win_size=10,
                 tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(
        weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(
        descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)]
    return top_ids


def random_sampling(weighted_scores):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids
