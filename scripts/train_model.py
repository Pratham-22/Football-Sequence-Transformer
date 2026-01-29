import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
import torch.optim as optim

from model.transformer import MaskedEventTransformer
from training.dataset import encode, valid_slice_flag, MLMDataset
from training.loss import mlm_loss

WINDOW = 40
EPOCHS = 10
BATCH = 16
MASK_PROB = 0.15
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    train = pd.read_csv("train.csv")
    valid = pd.read_csv("valid.csv")

    # scale
    scaler = MinMaxScaler()
    train.iloc[:, :] = scaler.fit_transform(train)
    valid.iloc[:, :] = scaler.transform(valid)

    ACTION_CLASSES = np.array(sorted(train["act"].unique()))
    ZONE_CLASSES = np.array(sorted(train["zone"].unique()))

    action_mask = len(ACTION_CLASSES)
    zone_mask = len(ZONE_CLASSES)

    weight_action = torch.tensor(
        compute_class_weight("balanced", ACTION_CLASSES, train["act"]),
        dtype=torch.float32
    )
    weight_zone = torch.tensor(
        compute_class_weight("balanced", ZONE_CLASSES, train["zone"]),
        dtype=torch.float32
    )

    train = valid_slice_flag(train, WINDOW)
    valid = valid_slice_flag(valid, WINDOW)

    enc_train = encode(train)
    enc_valid = encode(valid)

    train_ds = MLMDataset(train, enc_train, WINDOW, MASK_PROB, action_mask, zone_mask)
    valid_ds = MLMDataset(valid, enc_valid, WINDOW, MASK_PROB, action_mask, zone_mask)

    model = MaskedEventTransformer(
        d_model=len(ACTION_CLASSES)+len(ZONE_CLASSES)+6,
        hidden=1024,
        heads=1,
        act_in=len(ACTION_CLASSES)+1,
        act_out=len(ACTION_CLASSES),
        zone_in=len(ZONE_CLASSES)+1,
        zone_out=len(ZONE_CLASSES),
        cont_in=6,
        cont_out=6
    ).to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=1e-3)

    for e in range(EPOCHS):
        for X, Y, mask in DataLoader(train_ds, BATCH, shuffle=True):
            X, Y, mask = X.to(DEVICE), Y.to(DEVICE), mask.to(DEVICE)
            loss = mlm_loss(Y, model(X), mask, DEVICE, weight_action, weight_zone)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {e} done")

    torch.save(model.state_dict(), "masked_event_model.pt")

if __name__ == "__main__":
    main()
