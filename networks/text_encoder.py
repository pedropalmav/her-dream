from torch import nn

class TextEncoderGRU(nn.Module):
    """
    Encode one-hot character sequences into a stochastic latent logit (S, K).
    Input:  (B, T, L, V) float32 one-hot
    Output: (B, T, S, K) logit — misma forma que post_logit del RSSM
    """
    def __init__(self, vocab_size, embed_dim, hidden, stoch, discrete, num_layers=1):
        super().__init__()
        self.stoch = stoch
        self.discrete = discrete
        
        # Proyecta cada carácter one-hot a un embedding denso
        self.char_proj = nn.Linear(vocab_size, embed_dim, bias=False)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.RMSNorm(hidden, eps=1e-4),
            nn.ELU(),
            nn.Linear(hidden, stoch * discrete),
        )

    def forward(self, tokens):
        # tokens: (B, T, L, V) float one-hot
        leading = tokens.shape[:-2]
        L, V = tokens.shape[-2], tokens.shape[-1]
        flat = tokens.reshape(-1, L, V)              # (B*T, L, V)

        emb = self.char_proj(flat)                   # (B*T, L, embed_dim)

        _, h_n = self.gru(emb)                       # h_n: (num_layers, B*T, hidden)
        h_final = h_n[-1]                            # (B*T, hidden) última capa

        logit = self.head(h_final)                   # (B*T, S*K)
        return logit.reshape(*leading, self.stoch, self.discrete)