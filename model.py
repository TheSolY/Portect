import torch
from torch import nn
import lpips


class PortectModel(nn.Module):
    def __init__(self, phi, alpha=0.1, p=0.05, input_size=(640, 640)):
        super(PortectModel, self).__init__()
        self.input_size = input_size
        self.delta = torch.rand(input_size)
        self.phi = phi
        self.alpha = alpha
        self.p = p
        self.lpips_fn = lpips.LPIPS(net='vgg')

    def forward(self, inputs, targets):
        emb_targets = self.phi(targets)
        emb_inputs = self.phi(inputs + self.delta)
        loss = (torch.linalg.vector_norm(emb_targets - emb_inputs) ** 2 +
                self.alpha * torch.maximum(self.lpips_fn(inputs, inputs + self.delta) - self.p, torch.tensor(0.0)))
        return loss.mean()

