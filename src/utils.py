import torch

from torch.nn import L1Loss
from sklearn.metrics import f1_score, r2_score


def squared_hinge_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    """ref: https://www.tensorflow.org/api_docs/python/tf/keras/losses/SquaredHinge
    """
    hinge_loss = torch.maximum(1 - y_true * y_pred, torch.zeros_like(y_true))
    squared_loss = torch.square(hinge_loss)
    return squared_loss


def tanh_acc(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred_mapped = torch.where(y_pred > 0, 1, -1)
    mask = y_pred_mapped == y_true
    acc = mask.float().mean().item()
    return acc


def tanh_f1(y_pred: torch.Tensor, y_true: torch.Tensor, positive=1):
    y_pred_mapped = torch.where(y_pred > 0, 1, -1)
    score = f1_score(y_true.tolist(), y_pred_mapped.tolist(), pos_label=positive)
    return score


@torch.no_grad()
def mae_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    mae_loss = L1Loss()
    score = mae_loss(y_pred, y_true)
    return score.item()


@torch.no_grad()
def relative_mae_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    mae_loss = L1Loss(reduction='none')
    score = (mae_loss(y_pred, y_true) / torch.abs(y_true)).mean()
    return score.item()


@torch.no_grad()
def eval_r2_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    score = r2_score(y_true.tolist(), y_pred.tolist())
    return score
