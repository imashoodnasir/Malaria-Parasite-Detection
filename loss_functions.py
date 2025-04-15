import torch
import torch.nn as nn
import torch.nn.functional as F

def margin_loss(v_length, labels, m_plus=0.9, m_minus=0.1, lambda_=0.5):
    left = F.relu(m_plus - v_length)**2
    right = F.relu(v_length - m_minus)**2
    loss = labels * left + lambda_ * (1.0 - labels) * right
    return loss.sum(dim=1).mean()

def focal_loss(preds, targets, alpha=0.25, gamma=2.0):
    pt = torch.where(targets == 1, preds, 1 - preds)
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-12)
    return loss.sum(dim=1).mean()

def offset_loss(true_offsets, pred_offsets):
    diff = torch.abs(true_offsets - pred_offsets)
    loss = torch.where(diff < 1, 0.5 * diff**2, diff - 0.5)
    return loss.mean()

def reconstruction_loss(original, reconstructed):
    return F.mse_loss(reconstructed, original)

def total_loss(v_length, labels, x, recon_x, alpha1=1.0, alpha2=0.0005, alpha3=1.0, alpha4=1.0, focal=False, preds=None, targets=None, offsets_true=None, offsets_pred=None):
    l_margin = margin_loss(v_length, labels)
    l_recon = reconstruction_loss(x, recon_x)
    l_class = focal_loss(preds, targets) if focal else 0
    l_offset = offset_loss(offsets_true, offsets_pred) if offsets_true is not None else 0
    return alpha1 * l_margin + alpha2 * l_recon + alpha3 * l_class + alpha4 * l_offset
