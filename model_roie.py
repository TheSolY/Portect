import torch
from torch import nn, sigmoid
import lpips
from lpips import LPIPS
from utils import FeatureExtractor
import torch.nn.functional as F

feature_extractor = FeatureExtractor()
lpips = LPIPS()

def phi(img):
    return torch.tensor(feature_extractor.extract_features(img))

class PortectModel(nn.Module):
    def __init__(self, input_size):
        super(PortectModel, self).__init__()
        self.phi = phi
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5, stride=1, padding=0), nn.ReLU(), nn.AvgPool2d(kernel_size=2, stride=2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((40,40))
        self.conv3 = nn.Flatten() 
        self.fc0 = nn.Sequential(nn.Linear(40*40*16, 1024), nn.LayerNorm(1024), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU()) # This line is correct
        self.fc3 = nn.Sequential(nn.Linear(256, input_size[0]*input_size[1]), nn.LayerNorm(input_size[0]*input_size[1]), nn.Sigmoid())


    def forward(self, initial_input):
        temp = self.conv1(initial_input)
        temp = self.conv2(temp)
        temp = self.adaptive_pool(temp)
        temp = self.conv3(temp)
        temp = self.fc0(temp)
        temp = self.fc1(temp)
        temp = self.fc2(temp)
        beta = F.relu(self.fc3(temp))
        return beta

def PortectLoss(x, x_swapped, x_perturbed, alpha, p):
    phi_x = phi(x)
    phi_x_swapped = phi(x_swapped)
    phi_x_perturbed = phi(x_perturbed)
    identity_loss = torch.norm(phi_x_swapped - phi_x_perturbed, p=2)

    similarity_loss = torch.relu(lpips(x/255, x_perturbed/255) - p)

    total_loss = identity_loss + alpha * similarity_loss
    return total_loss, identity_loss, similarity_loss




