import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from model import PortectAdditiveModel, PortectLoss
from utils import FeatureExtractor

# the root that contains the data in the expected format
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

optimizer = torch.optim.SGD(model.params, lr=0.1)
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(org_image)
    loss = loss_fn(org_image, outputs, target_image)
    loss.backward()
    optimizer.step()
    print(loss.item())

torch.save(model.delta, './delta')
