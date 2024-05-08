import torch
from torch import nn
import lpips
from lpips import LPIPS
from utils import FeatureExtractor

class PortectModel(nn.Module):
    def __init__(self, input_size=(640,640)):
        super(PortectModel, self).__init__()
        self.delta_x = torch.nn.Parameter(torch.rand((640,640), requires_grad=True))
        self.params = torch.nn.ParameterList([torch.nn.Parameter(self.delta_x)])

    def forward(self, x):
        x_additive = x + self.delta_x
        delta_x = self.delta_x
        return x_additive, delta_x

def PortectLoss(x, x_additive, x_swapped, delta_x, phi, alpha = 0.5, p=0.1):
  percept = LPIPS()

  images_norm_term = torch.norm(phi(x_swapped) - phi(x_additive))
  regularization_term = alpha * torch.max(percept(x, x_additive) - p, torch.tensor(0.0))
  
  loss = images_norm_term + regularization_term
  return loss




