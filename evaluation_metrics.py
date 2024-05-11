from diffusers.utils import load_image
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize

from utils import FeatureExtractor

IMAGE_SIZE = (640, 640)
feature_extractor = FeatureExtractor()
resize = Resize(IMAGE_SIZE)

def compute_face_embedding(image_path):
    img = read_image(image_path, mode=ImageReadMode.RGB)
    img = resize(img)
    img = img.float()
    return torch.tensor(feature_extractor.extract_features(img))


def compute_avg_embedding(images_path):
  images_embedding = []
  for filename in os.listdir(images_path):
    image_path = os.path.join(images_path, filename)
    try:
      single_embedding = compute_face_embedding(image_path)
      images_embedding.append(single_embedding)
    except Exception as e:
      print(f'Could not retrieve embedding for {image_path}')
      continue
  images_embedding = np.stack(images_embedding)
  images_embedding_centroid = images_embedding.mean(axis=0)
  return torch.tensor(images_embedding_centroid)


def identity_score_matching(generated_image_path, identity_embedding):
    try:
      image_emb = compute_face_embedding(generated_image_path)
    except Exception as e:
      return -1
    identity_score = F.cosine_similarity(image_emb, identity_embedding, dim=0)
    return identity_score


def calculate_FDFR_and_ISM(generated_images_path, original_images_path):
    image_list = os.listdir(generated_images_path)
    fail_detection_count = 0
    total_ism = 0
    avg_embedding = compute_avg_embedding(original_images_path) 

    for image_name in image_list:
        image_path = os.path.join(generated_images_path, image_name)
        ism = identity_score_matching(image_path, avg_embedding)
        if ism==-1:
            fail_detection_count += 1
        else:
            total_ism += ism
    if fail_detection_count == len(image_list): # all generated images failed to show identity in them
      avg_ism = None
      fail_detection_ratio = 1
    else: 
      avg_ism = total_ism/(len(image_list)-fail_detection_count)
      fail_detection_ratio = fail_detection_count/len(image_list)
    return avg_ism, fail_detection_ratio


avg_ism, fail_detection_ratio = calculate_FDFR_and_ISM('/content/Portect/assets/swapped_images', '/content/Portect/assets/org_images')
print(avg_ism, fail_detection_ratio)








