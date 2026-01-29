import torch
import numpy as np
import random

OTHER_COLS = [
    'zone_s','zone_deltay','zone_deltax','zone_sg','zone_thetag'
]

def encode(df):
    act = torch.tensor(df["act"].values, dtype=torch.long)
    zone = torch.tensor(df["zone"].values, dtype=torch.long)
    dT = torch.tensor(df["deltaT"].values, dtype=torch.float32)
    other = torch.tensor(df[OTHER_COLS].values, dtype=torch.float32)

    return torch.cat([act.view(-1,1), zone.view(-1,1), dT.view(-1,1), other], dim=1)

def valid_slice_flag(df, window):
    df = df.copy()
    df["valid_slice_flag"] = True

    for i in range(len(df) - 1):
        if df.loc[i, "MID"] != df.loc[i+1, "MID"]:
            df.loc[i+1-window:i+1, "valid_slice_flag"] = False

    df.loc[len(df)-window:, "valid_slice_flag"] = False
    return df

class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, df, encoded, win, mask_prob, action_mask, zone_mask):
        self.df = df.reset_index(drop=True)
        self.enc = encoded
        self.win = win
        self.mask_prob = mask_prob
        self.action_mask = action_mask
        self.zone_mask = zone_mask
        self.valid_idx = np.where(self.df["valid_slice_flag"])[0]

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        j = self.valid_idx[idx]
        X = self.enc[j:j+self.win].clone()
        Y = X.clone()

        mask = torch.zeros(self.win, dtype=torch.bool)
        num_mask = max(1, int(self.win * self.mask_prob))
        positions = random.sample(range(self.win), num_mask)

        for p in positions:
            mask[p] = True
            X[p,0] = self.action_mask
            X[p,1] = self.zone_mask
            X[p,2] = 0.0

        return X.float(), Y.float(), mask
