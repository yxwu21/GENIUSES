import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from src.utils import squared_hinge_loss


class SquaredHingeLoss(nn.Module):
    def __init__(self):
        super(SquaredHingeLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = squared_hinge_loss(y_pred, y_true)
        return loss.mean()


class PerceptronLoss(nn.Module):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        true_sign = torch.where(y_true > self.threshold, 1., -1.)
        pred_sign = y_pred - self.threshold
        sign_loss = torch.max(torch.zeros_like(pred_sign), torch.abs(y_true) - pred_sign * true_sign)
        return sign_loss.mean()


class MLSESModel(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super().__init__()
        self.layer1 = nn.Linear(dim1, dim2)
        self.layer2 = nn.Linear(dim2, dim3)
        self.layer3 = nn.Linear(dim3, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x

    def load_ckpt(self, ckpt_file):
        ckpt = json.load(open(ckpt_file))

        with torch.no_grad():
            for i in range(1, 4):
                layer: nn.Linear = getattr(self, f'layer{i}')
                layer.weight.copy_(
                    layer.weight.new_tensor(
                        ckpt[f'W{i}']
                    )
                )
                layer.bias.copy_(
                    layer.bias.new_tensor(
                        ckpt[f'b{i}']
                    )
                )


class RefinedMLSESModel(nn.Module):
    def __init__(self, dim1, dim2, dim3, probe_radius):
        super().__init__()
        self.layer1 = nn.Linear(dim1, dim2)
        self.layer2 = nn.Linear(dim2, dim3)
        self.layer3 = nn.Linear(dim3, 1)

        self.probe_radius = probe_radius

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x
