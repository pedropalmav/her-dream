import torch.nn.functional as F
from torch import nn


class TextEncoderGRU(nn.Module):
    """
    Encode character-id sequences into a stochastic latent logit (S, K).
    Input:  (B, T, L) integer token ids in [0, V-1]
    Output: (B, T, S, K) logit — misma forma que post_logit del RSSM
    """

    def __init__(self, config, stoch, discrete, act):
        super().__init__()
        self.stoch = stoch
        self.discrete = discrete
        self.vocab_size = int(config.vocab_size)

        # Proyecta cada carácter one-hot a un embedding denso
        self.char_proj = nn.Linear(config.vocab_size, config.embed_dim, bias=False)
        act = getattr(nn, act)
        self.gru = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=config.hidden,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.RMSNorm(config.hidden, eps=1e-4),
            act(),
            nn.Linear(config.hidden, stoch * discrete),
        )

    def forward(self, tokens):
        # tokens: (B, T, L) integer ids. One-hot is materialised here to keep
        # the replay buffer small (ids are stored as int8).
        leading = tokens.shape[:-1]
        L = tokens.shape[-1]
        flat = tokens.reshape(-1, L).long()  # (B*T, L)
        one_hot = F.one_hot(flat, num_classes=self.vocab_size).to(self.char_proj.weight.dtype)

        emb = self.char_proj(one_hot)  # (B*T, L, embed_dim)

        _, h_n = self.gru(emb)  # h_n: (num_layers, B*T, hidden)
        h_final = h_n[-1]  # (B*T, hidden) última capa

        logit = self.head(h_final)  # (B*T, S*K)
        return logit.reshape(*leading, self.stoch, self.discrete)
