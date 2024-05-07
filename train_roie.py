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

feature_extractor = FeatureExtractor()


def phi(img):
    return torch.tensor(feature_extractor.extract_features(img))


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

optimizer = torch.optim.SGD(model.params, lr=0.0001)

for i, data in enumerate(train_loader):
    if (i==0):
      continue
    x, x_swapped = data
    print(f"image #{i}")
    for epoch in range(EPOCHS):
        loss = 0
        optimizer.zero_grad()
        x_additive, delta_x = model.forward(x)
        try:
            with suppress_stdout():
              loss = PortectLoss(x, x_additive, x_swapped, delta_x, phi)
            loss.backward()
        except IndexError:
            print(f"error for image {i}")

        optimizer.step()

        print(f"\tepoch #{epoch}, loss: {loss.item()}")

torch.save(model.delta_x, './delta')
