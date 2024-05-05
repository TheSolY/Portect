import os
from utils import FaceSwapper2, farthest_neighbor, all_image_path_from_index
from diffusers.utils import load_image
import numpy as np
import torch

# the data should be in a folder called "org_images" inside the root directory
images_root = './assets/'
celeba_path = '/datasets/celeba'

org_images_dir = os.path.join(images_root, 'org_images')
if not os.path.exists(org_images_dir):
    raise FileNotFoundError(f'{org_images_dir} not found')

swapped_imaged_dir = os.path.join(images_root, 'swapped_images')

if os.path.exists(swapped_imaged_dir):
    raise FileExistsError(f'{swapped_imaged_dir} already exists')
else:
    os.mkdir(swapped_imaged_dir)

# Rename to indices for easy matching for the training later
for i, image in enumerate(os.listdir(org_images_dir)):
    _, file_extention = os.path.splitext(image)
    if file_extention in ('.jpg', '.jpeg', '.png'):
        os.rename(os.path.join(org_images_dir, image), os.path.join(org_images_dir, str(i) + file_extention))

face_swapper = FaceSwapper2()

# Calculate the embedding centroid of the original images and find the target identity
org_images_emb = []
for filename in os.listdir(org_images_dir):
    image = load_image(os.path.join(org_images_dir, filename))
    emb = face_swapper.extract_features(image)
    org_images_emb.append(emb)

org_images_emb = np.stack(org_images_emb)
org_images_emb_centroid = org_images_emb.mean(axis=0)

id_centroids = torch.load('assets/celeba_id_embedding_centroids')
id_identity_idxs = torch.load('assets/unique_ids')
idx, vec = farthest_neighbor(org_images_emb_centroid, id_centroids.numpy())
# identity_idx = id_identity_idxs[idx].item()

# Create the target (face swapped) images
# Later replace the image for image with the embedding centroid, when we can face swap from embedding
# target_id_filenames = all_image_path_from_index(identity_idx,  os.path.join(celeba_path, 'identity_CelebA.txt'))

for image_filename in os.listdir(org_images_dir):
    _, file_extention = os.path.splitext(image_filename)
    if file_extention in ('.jpg', '.jpeg', '.png'):
        face_swapper.swap_face(os.path.join(org_images_dir, image_filename),
                               vec,
                               os.path.join(swapped_imaged_dir, image_filename))







