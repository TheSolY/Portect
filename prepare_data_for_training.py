import os
import cv2
from utils import FaceSwapper2, farthest_neighbor, interpolate_embedding, resize_img
from diffusers.utils import load_image
import numpy as np
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# the data should be in a folder called "src_images" inside the root directory
images_root = './training_images/'
celeba_path = '/datasets/celeba'

src_images_dir = os.path.join(images_root, 'src_images')
if not os.path.exists(src_images_dir):
    raise FileNotFoundError(f'{src_images_dir} not found')

org_image_dir = os.path.join(images_root, 'org_images')
swapped_image_dir = os.path.join(images_root, 'swapped_images')

if os.path.exists(swapped_image_dir) or os.path.exists(org_image_dir):
    raise FileExistsError(f'{swapped_image_dir} or {org_image_dir} already exists')
else:
    os.mkdir(swapped_image_dir)
    os.mkdir(org_image_dir)

face_swapper = FaceSwapper2()

# Resize the images and save by index for order matching during training,
# Calculate the embedding centroid of the original images and find the target identity
org_images_emb = []
for i, filename in enumerate(os.listdir(src_images_dir)):
    _, file_extention = os.path.splitext(filename)
    if file_extention in ('.jpg', '.jpeg', '.png', '.JPG'):
        image = load_image(os.path.join(src_images_dir, filename))
        image = resize_img(image, 1024, 1024, pad_to_max_side=True)
        img_to_save = np.array(image.copy())[:, :, ::-1]
        cv2.imwrite(os.path.join(org_image_dir, str(i) + file_extention), img_to_save)

        emb = face_swapper.extract_features(image)
        org_images_emb.append(emb)

org_images_emb = np.stack(org_images_emb)
org_images_emb_centroid = org_images_emb.mean(axis=0)

id_centroids = torch.load('assets/celeba_id_embedding_centroids')
id_identity_idxs = torch.load('assets/unique_ids')
idx, vec = farthest_neighbor(org_images_emb_centroid, id_centroids.numpy())

target_emb = interpolate_embedding(org_images_emb, vec, 0.05)

# Create the target (face swapped) images
for i, image_filename in enumerate(os.listdir(org_image_dir)):
    _, file_extention = os.path.splitext(image_filename)
    if file_extention in ('.jpg', '.jpeg', '.png', '.JPG'):
        face_swapper.swap_face(os.path.join(org_image_dir, image_filename),
                               target_emb[i],
                               os.path.join(swapped_image_dir, image_filename))







