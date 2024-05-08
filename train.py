import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from model import PortectAdditiveModel, PortectLoss
from utils import FeatureExtractor

# the root that contains the data in the expected format
# data_root = './assets'
EPOCHS = 10
org_image_path = './assets/org_images/6.jpg'
target_image_path = './assets/swapped_images/6.jpg'

IMAGE_SIZE = (640, 640)
resize = Resize(IMAGE_SIZE)

org_image = resize(read_image(org_image_path))
target_image = resize(read_image(target_image_path))



feature_extractor = FeatureExtractor()
bounding_box = feature_extractor.extract_bounding_box(org_image)
bounding_box_int = [int(b) for b in bounding_box]
print(bounding_box)

model = PortectAdditiveModel(bounding_box_int)


def phi(img):
    return torch.tensor(feature_extractor.extract_features(img))


loss_fn = PortectLoss(bounding_box=bounding_box_int, phi=phi)
# def load_and_prep_image(img_path):
#     img = read_image(img_path, mode=ImageReadMode.RGB)
#     img = resize(img)
#     img = img / 255.0
#     return img

# org_images_dir = os.path.join(data_root, 'org_images')
# target_images_dir = os.path.join(data_root, 'swapped_images')
#
# num_images = len(os.listdir(org_images_dir))
# print(f'number of samples: {num_images}')
#
# org_images = []
# target_images = []

# Need to make sure that the datasets are in the same order but the extension is determined by the target face,
# so thy should have the same file name beside the extension.

# resize = Resize(IMAGE_SIZE)
#
# for filename in sorted(os.listdir(org_images_dir)):
#     _, file_extention = os.path.splitext(filename)
#     if file_extention in ('.jpg', '.jpeg', '.png'):
#         # img = load_and_prep_image(os.path.join(org_images_dir, filename))
#         img = read_image(os.path.join(org_images_dir, filename), mode=ImageReadMode.RGB)
#         img = resize(img)
#         org_images.append(img)
#
# for filename in sorted(os.listdir(target_images_dir)):
#     _, file_extention = os.path.splitext(filename)
#     if file_extention in ('.jpg', '.jpeg', '.png'):
#         # img = load_and_prep_image(os.path.join(target_images_dir, filename))
#         img = read_image(os.path.join(target_images_dir, filename), mode=ImageReadMode.RGB)
#         img = resize(img)
#         target_images.append(img)
#
# org_images = torch.stack(org_images)
# target_images = torch.stack(target_images)
#
# train_dataset = TensorDataset(org_images, target_images)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.SGD(model.params, lr=0.1)
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(org_image)
    loss = loss_fn(org_image, outputs, target_image)
    loss.backward()
    optimizer.step()
    print(loss.item())

# for epoch in range(EPOCHS):
#     loss = 0
#     running_loss = 0
#     for i, data in enumerate(train_loader):
#         inputs, targets = data
#
#         optimizer.zero_grad()
#
#         outputs = model.forward(inputs)
#         try:
#             loss = loss_fn(inputs, outputs, targets)
#             loss.backward()
#         except IndexError:
#             print(i)
#
#         optimizer.step()
#
#         running_loss += loss.item()
#     running_loss /= len(train_loader)
#     print(running_loss)

torch.save(model.delta, './delta')
