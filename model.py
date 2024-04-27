import torch
from torch import nn
import lpips


class PortectModel(nn.Module):
    def __init__(self, input_size=(640, 640)):
        super(PortectModel, self).__init__()
        self.delta = torch.rand(input_size, requires_grad=True)
        self.params = torch.nn.ParameterList([torch.nn.Parameter(self.delta)])

    def forward(self, inputs):
        outputs = torch.nn.Identity()(inputs + self.delta)
        return outputs


class PortectLoss(nn.Module):
    def __init__(self, phi, alpha=0.1, p=0.05):
        super(PortectLoss, self).__init__()
        self.phi = phi
        self.alpha = alpha
        self.p = p
        self.lpips_fn = lpips.LPIPS(net='alex')

    def forward(self, inputs, outputs, targets):
        emb_targets = self.phi(targets)
        emb_inputs = self.phi(outputs.clone().detach())
        loss = (torch.linalg.vector_norm(emb_targets - emb_inputs) ** 2 +
                self.alpha * torch.maximum(self.lpips_fn(inputs, outputs) - self.p, torch.tensor(0.0)))
        return loss.mean()
