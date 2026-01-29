import torch
from torch import nn

def mlm_loss(Y, preds, mask, device, weight_action, weight_zone):
    d_pred, z_pred, a_pred = preds

    y_act = Y[:,:,0].long()
    y_zone = Y[:,:,1].long()
    y_dT = Y[:,:,2].float()

    mask_flat = mask.view(-1)

    act_p = a_pred.reshape(-1, len(weight_action))[mask_flat]
    zone_p = z_pred.reshape(-1, len(weight_zone))[mask_flat]
    dT_p = d_pred.reshape(-1)[mask_flat]

    act_t = y_act.reshape(-1)[mask_flat]
    zone_t = y_zone.reshape(-1)[mask_flat]
    dT_t = y_dT.reshape(-1)[mask_flat]

    ce_action = nn.CrossEntropyLoss(weight=weight_action.to(device))(act_p, act_t)
    ce_zone = nn.CrossEntropyLoss(weight=weight_zone.to(device))(zone_p, zone_t)
    rmse = ((dT_p - dT_t) ** 2).mean().sqrt()

    return ce_action + ce_zone + rmse
