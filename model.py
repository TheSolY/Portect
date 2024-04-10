import torch
from torch import nn
import lpips


class PortectModel(nn.Module):
    def __init__(self, phi, omega, alpha=0.1, p=0.05, input_size=(640, 640)):
        super(PortectModel, self).__init__()
        self.input_size = input_size
        self.delta = torch.rand(input_size)
        self.phi = phi
        self.omega = omega
        self.alpha = alpha
        self.p = p
        self.lpips_fn = lpips.LPIPS(net='vgg')

    def forward(self, x):
        return x + self.delta

    def loss(self, inputs, targets):
        emb_targets = self.phi(inputs, targets)
        emb_inputs = self.phi(self.omega(self.forward(inputs)))
        loss = (torch.linalg.vector_norm(emb_targets - emb_inputs) ** 2 +
                self.alpha * torch.maximum(self.lpips_fn(self.delta) - self.p, torch.Tensor(0.0)))
        return loss.mean()

