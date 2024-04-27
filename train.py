import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from model import PortectModel, PortectLoss
from utils import FeatureExtractor

# the root that contains the data in the expected format
data_root = './assets'
EPOCHS = 10
IMAGE_SIZE = (640, 640)

feature_extractor = FeatureExtractor()

def phi(img):
    return torch.tensor(feature_extractor.extract_features(img))


model = PortectModel()
loss_fn = PortectLoss(phi=phi)

org_images_dir = os.path.join(data_root, 'org_images')
target_images_dir = os.path.join(data_root, 'swapped_images')

num_images = len(os.listdir(org_images_dir))
print(f'number of samples: {num_images}')

org_images = []
target_images = []

# Need to make sure that the datasets are in the same order but the extension is determined by the target face,
# so thy should have the same file name beside the extension.

resize = Resize(IMAGE_SIZE)

for filename in os.listdir(org_images_dir):
    _, file_extention = os.path.splitext(filename)
    if file_extention in ('.jpg', '.jpeg', '.png'):
        img = read_image(os.path.join(org_images_dir, filename), mode=ImageReadMode.RGB)
        img = resize(img)
        org_images.append(img)

for filename in os.listdir(target_images_dir):
    _, file_extention = os.path.splitext(filename)
    if file_extention in ('.jpg', '.jpeg', '.png'):
        img = read_image(os.path.join(target_images_dir, filename), mode=ImageReadMode.RGB)
        img = resize(img)
        target_images.append(img)

org_images = torch.stack(org_images)
target_images = torch.stack(target_images)

train_dataset = TensorDataset(org_images, target_images)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.SGD(model.params, lr=0.1)

for epoch in range(EPOCHS):
    loss = 0
    running_loss = 0
    for i, data in enumerate(train_loader):
        inputs, targets = data

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = loss_fn(inputs, outputs, targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    running_loss /= len(train_loader)
    print(running_loss)

torch.save(model.delta, './delta')
