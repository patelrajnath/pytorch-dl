from torch import nn


class TokenEmbedding(nn.Module):
    """
    """
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x):
        return self.token_emb(x)