import torch
from torch import nn
import lpips
from lpips import LPIPS
from utils import FeatureExtractor


feature_extractor = FeatureExtractor()


class PortectModel(nn.Module):
    def __init__(self, input_size=(640,640)):
        super(PortectModel, self).__init__()
        self.phi = phi
        self.delta_x = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )
        self.lpips = LPIPS()

    def forward(self, x, x_swapped):
        phi_x = self.phi(x)
        phi_x_swapped = self.phi(x_swapped)
        delta_x = self.delta_x(x)
        x_perturbed = x + delta_x
        phi_x_perturbed = self.phi(x_perturbed)
        lpips_score = self.lpips(x, x_perturbed)
        return phi_x, phi_x_swapped, phi_x_perturbed, lpips_score, x_perturbed

def PortectLoss(phi_x, phi_x_swapped, phi_x_perturbed, lpips_score, alpha = 0.5, p=0.05):
    identity_loss = torch.norm(phi_x_swapped - phi_x_perturbed, p=2)
    perceptual_loss = torch.relu(lpips_score - p)
    total_loss = identity_loss + alpha * perceptual_loss

    return total_loss


def phi(img):
    return torch.tensor(feature_extractor.extract_features(img))


