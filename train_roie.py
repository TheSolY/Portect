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
import copy

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
EPOCHS = 30
IMAGE_SIZE = (640, 640)

def load_and_prep_image(img_path):
    img = read_image(img_path, mode=ImageReadMode.RGB)
    img = resize(img)
    img = img / 255.0
    return img



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

feature_extractor = FeatureExtractor()

for i, data in enumerate(train_loader):
    if (i==0):
      continue
    x, x_swapped = data
    initial_input = torch.rand((1,1024))

    bounding_box = feature_extractor.extract_bounding_box(x)
    bounding_box_int = [int(b) for b in bounding_box]
    cloak_w = bounding_box_int[2] - bounding_box_int[0]
    cloak_h = bounding_box_int[3] - bounding_box_int[1]
    input_size = (cloak_w, cloak_h)
    row_slice = slice(bounding_box_int[0], bounding_box_int[0] + cloak_w)
    col_slice = slice(bounding_box_int[1], bounding_box_int[1] + cloak_h)
    model = PortectModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    x = x.float()
    x_swapped = x_swapped.float()
    initial_input = torch.cat((x[:,:,row_slice, col_slice], x_swapped[:,:,row_slice, col_slice]), dim=1)
    #initial_input.view(1, 6, cloak_w, cloak_h)
    print(f"image #{i}")
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        beta = model(initial_input) # shape of bounding box
        beta = beta.view(-1, input_size[0],input_size[1])

        x_perturbed = copy.deepcopy(x)
        x_perturbed[:,:,row_slice, col_slice] = torch.clip(beta * x[:,:,row_slice, col_slice] + (1 - beta) * x_swapped[:,:, row_slice, col_slice], 0, 255.0)
        try:
            with suppress_stdout():
              loss, identity_loss, similarity_loss = PortectLoss(x, x_swapped, x_perturbed, alpha = 0, p=0.05)
            loss.backward()
            # debug - print gradients
            #for name, param in model.named_parameters():
            #  print(name, param.requires_grad)
        except IndexError:
            print(f"error for image {i}")
            break
        optimizer.step()
        print(f"\tepoch #{epoch}, loss: {loss.item():.20f}, identity_loss: {identity_loss.item()}, similarity_loss: {similarity_loss.item()}")
    
    model.eval()

torch.save(model.delta_x, './delta')
