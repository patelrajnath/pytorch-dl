import torch
from torch import nn
import torch.nn.functional as F
from models.utils import models_util


class SelfAttention(nn.Module):
    """
    This is basic transformer model
    """
    def __init__(self, k, heads):
        super().__init__()
        self.k, self.heads = k, heads
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalue = nn.Linear(k, k * heads, bias=False)
        self.tokey = nn.Linear(k, k * heads, bias=False)
        self.unifyheads = nn.Linear(k*heads, k, bias=False)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        query = self.toqueries(x).view(b, t, h, k)
        key = self.tokey(x).view(b, t, h, k)
        value = self.tovalue(x).view(b, t, h, k)

        query = query.transpose(1, 2).contiguous().view(b*h, t, k)
        key = key.transpose(1, 2).contiguous().view(b*h, t, k)
        value = value.transpose(1, 2).contiguous().view(b*h, t, k)

        query = query / (k**(1/4))
        key = key / (k**(1/4))

        dot = torch.bmm(query, key.transpose(1, 2))
        dot = F.softmax(dot, dim=2)
        # print(torch.sum(dot, dim=2))
        out = torch.bmm(dot, value).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h*k)
        return self.unifyheads(out)
        # print('key:', key.size(), 'value:', value.size(), 'query:', query.size(),
        #       'dot', dot.size(), "out", out.size())


class TransformerBlock(nn.Module):
  def __init__(self, k, heads, ff=32, dropout=0.0):
    super().__init__()

    self.attention = SelfAttention(k, heads=heads)

    self.norm1 = nn.LayerNorm(k)
    self.norm2 = nn.LayerNorm(k)

    self.ff = nn.Sequential(
      nn.Linear(k, ff * k),
      nn.ReLU(),
      nn.Linear(ff * k, k))

    self.do = nn.Dropout(dropout)

  def forward(self, x):
    attended = self.attention(x)
    x = self.norm1(attended + x)
    x = self.do(x)
    fedforward = self.ff(x)
    x = self.norm2(fedforward + x)
    x = self.do(x)
    return x


class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes, dropout=0.0):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        # The sequence of transformer blocks that does all the
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        # generate position embeddings
        positions = torch.arange(t, device=models_util.d())
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        # Average-pool over the t dimension and project to class
        # probabilities
        x = self.toprobs(x.mean(dim=1))

        return F.log_softmax(x, dim=1)