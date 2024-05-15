import torch
from torch import nn
import lpips


class PortectAdditiveModel(nn.Module):
    def __init__(self, bounding_box, input_size=(640, 640)):
        super(PortectAdditiveModel, self).__init__()
        # RetinaNet bb format: x_min, y_min, x_max, y_max
        self.bb = bounding_box
        cloak_w = self.bb[2] - self.bb[0]
        cloak_h = self.bb[3] - self.bb[1]
        self.delta = torch.autograd.Variable(torch.rand((cloak_w, cloak_h)), requires_grad=True)
        self.params = torch.nn.ParameterList([torch.nn.Parameter(self.delta)])

    def forward(self, inputs):
        x = inputs.to(dtype=torch.float32)
        x[:, self.bb[0]:self.bb[2], self.bb[1]:self.bb[3]].add_(self.delta)
        outputs = torch.nn.Identity()(x)
        return torch.clip(outputs, 0, 255.0)


class PortectModel(nn.Module):
    def __init__(self):
        super(PortectModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x / 255.0
        x = self.encoder(x)
        x = self.decoder(x)
        x = x * 255.0
        return x


class PortectLoss(nn.Module):
    def __init__(self, bounding_box, phi, alpha=0.1, p=0.05):
        super(PortectLoss, self).__init__()
        self.phi = phi
        self.alpha = alpha
        self.p = p
        self.lpips_fn = lpips.LPIPS(net='alex')
        self.bb = bounding_box

    def forward(self, inputs, outputs, targets):
        emb_targets = self.phi(targets)
        emb_inputs = self.phi(outputs.clone().detach().type(torch.uint8))
        loss = (torch.linalg.vector_norm(emb_targets - emb_inputs) ** 2 +
                self.alpha * torch.maximum(self.lpips_fn(
                    inputs[:, self.bb[0]:self.bb[2], self.bb[1]:self.bb[3]],
                    outputs[:, self.bb[0]:self.bb[2], self.bb[1]:self.bb[3]])
                                           - self.p, torch.tensor(0.0)))
        return loss.mean()
