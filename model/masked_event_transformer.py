# model/masked_event_transformer.py
import torch
from torch import nn

def positional_encoding(L: int, D: int, device):
    pos = torch.arange(L, device=device).float().unsqueeze(1)       # [L,1]
    i = torch.arange(D, device=device).float().unsqueeze(0)         # [1,D]
    angle = pos / (10000 ** (2 * (i // 2) / D))
    pe = torch.zeros((L, D), device=device)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe.unsqueeze(0)  # [1,L,D]

class MaskedEventTransformer(nn.Module):
    """
    Input X shape: [B, L, F]
      X[:,:,0] = action_id (long)
      X[:,:,1] = zone_id   (long)
      X[:,:,2:] = continuous features (float)

    Outputs:
      deltaT_pred: [B, L, 1]
      zone_logits: [B, L, Z]
      action_logits: [B, L, A]
      hidden: [B, L, d_model]
    """
    def __init__(
        self,
        d_model: int,
        hidden: int,
        heads: int,
        act_in: int,
        act_out: int,
        zone_in: int,
        zone_out: int,
        cont_in: int,
        cont_out: int,
    ):
        super().__init__()
        self.emb_act = nn.Embedding(act_in, act_out)
        self.emb_zone = nn.Embedding(zone_in, zone_out)
        self.lin0 = nn.Linear(cont_in, cont_out)

        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            batch_first=True,
            dim_feedforward=hidden
        )

        self.lin_deltaT = nn.Linear(d_model, 1)
        self.lin_zone = nn.Linear(d_model, zone_out)
        self.lin_action = nn.Linear(d_model, act_out)

    def forward(self, X, return_hidden=False):
        B, L, F = X.shape

        a = self.emb_act(X[:, :, 0].long())
        z = self.emb_zone(X[:, :, 1].long())
        c = self.lin0(X[:, :, 2:].float())

        Xcat = torch.cat([a, z, c], dim=2)     # [B,L,d_model]
        pe = positional_encoding(L, Xcat.size(2), X.device)
        hidden = self.encoder(Xcat + pe)

        out = (
            self.lin_deltaT(hidden),
            self.lin_zone(hidden),
            self.lin_action(hidden),
        )
        if return_hidden:
            return (*out, hidden)
        return out
