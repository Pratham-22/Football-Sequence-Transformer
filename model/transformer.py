import torch
from torch import nn

def positional_encoding(L, D, device):
    pos = torch.arange(L, device=device).float().unsqueeze(1)
    i = torch.arange(D, device=device).float().unsqueeze(0)
    angle = pos / (10000 ** (2 * (i // 2) / D))

    pe = torch.zeros((L, D), device=device)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe.unsqueeze(0)

class MaskedEventTransformer(nn.Module):
    def __init__(
        self, d_model, hidden, heads,
        act_in, act_out,
        zone_in, zone_out,
        cont_in, cont_out
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
        a = self.emb_act(X[:, :, 0].long())
        z = self.emb_zone(X[:, :, 1].long())
        c = self.lin0(X[:, :, 2:].float())

        Xcat = torch.cat([a, z, c], dim=2)
        pe = positional_encoding(Xcat.size(1), Xcat.size(2), X.device)
        hidden = self.encoder(Xcat + pe)

        if return_hidden:
            return (
                self.lin_deltaT(hidden),
                self.lin_zone(hidden),
                self.lin_action(hidden),
                hidden
            )

        return (
            self.lin_deltaT(hidden),
            self.lin_zone(hidden),
            self.lin_action(hidden)
        )
