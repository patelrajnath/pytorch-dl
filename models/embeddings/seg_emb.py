from torch import nn


class SegmentEmbedding(nn.Module):
    """

    """
    def __init__(self, labels, emb_dim):
        super().__init__()
        self.seg_emb = nn.Embedding(labels, emb_dim)

    def forward(self, x):
        self.seg_emb(x)