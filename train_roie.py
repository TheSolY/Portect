import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from model_roie import PortectModel, PortectLoss
from utils import FeatureExtractor
import sys
import os
from contextlib import contextmanager
from lpips import LPIPS

# This context manager captures stdout
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# the root that contains the data in the expected format
data_root = './assets'
EPOCHS = 10
IMAGE_SIZE = (640, 640)

def load_and_prep_image(img_path):
    img = read_image(img_path, mode=ImageReadMode.RGB)
    img = resize(img)
    img = img / 255.0
    return img


model = PortectModel()

org_images_dir = os.path.join(data_root, 'org_images')
target_images_dir = os.path.join(data_root, 'swapped_images')

num_images = len(os.listdir(org_images_dir))
num_images_target = len(os.listdir(target_images_dir))


org_images = []
target_images = []

# Need to make sure that the datasets are in the same order but the extension is determined by the target face,
# so thy should have the same file name beside the extension.

resize = Resize(IMAGE_SIZE)

for filename in sorted(os.listdir(org_images_dir)):
    _, file_extention = os.path.splitext(filename)
    if file_extention in ('.jpg', '.jpeg', '.png'):
        # img = load_and_prep_image(os.path.join(org_images_dir, filename))
        img = read_image(os.path.join(org_images_dir, filename), mode=ImageReadMode.RGB)
        img = resize(img)
        org_images.append(img)

for filename in sorted(os.listdir(target_images_dir)):
    _, file_extention = os.path.splitext(filename)
    if file_extention in ('.jpg', '.jpeg', '.png'):
        # img = load_and_prep_image(os.path.join(target_images_dir, filename))
        img = read_image(os.path.join(target_images_dir, filename), mode=ImageReadMode.RGB)
        img = resize(img)
        target_images.append(img)

print(f'number of original images: {len(org_images)}')
print(f'number of swapped images: {len(target_images)}')

org_images = torch.stack(org_images)
target_images = torch.stack(target_images)

train_dataset = TensorDataset(org_images, target_images)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for i, data in enumerate(train_loader):
    if (i==0):
      continue
    x, x_swapped = data
    x = x.float()
    x_swapped = x_swapped.float()
    print(f"image #{i}")
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        phi_x, phi_x_swapped, phi_x_perturbed, lpips_score, x_perturbed = model(x, x_swapped)
        try:
            with suppress_stdout():
              loss = PortectLoss(phi_x, phi_x_swapped, phi_x_perturbed, lpips_score)
            loss.backward()
        except IndexError:
            print(f"error for image {i}")
        optimizer.step()
        # Print gradients
        #for name, param in model.named_parameters():
            #if param.requires_grad:
                 #print(name, param.grad)
        #print(f"x-x_additive: {x-x_additive}")
        #print(f"x-x_additive max value: {torch.max(torch.abs(x-x_additive))}")
        print(f"\tepoch #{epoch}, loss: {loss.item()}, lpips_score: {lpips_score.item()}")
    
    model.eval()

torch.save(model.delta_x, './delta')
