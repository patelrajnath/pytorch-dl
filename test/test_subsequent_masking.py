import torch

from dataset.iwslt_data import subsequent_mask
from torch.autograd import Variable


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


scores = torch.tensor([0.2, 0.3, 0.3, 0.4, 0.3])
mask = make_std_mask(torch.tensor([48, 45, 67, 36, 49]), 0)
print(mask)
scores = scores.masked_fill(mask == 0, -1e9)
print(scores)
