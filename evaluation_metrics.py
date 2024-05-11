from diffusers.utils import load_image
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from evaluations.backbones.iresnet import iresnet18
from pathlib import Path
from torchvision import io
from torchvision import transforms

from utils import FeatureExtractor

IMAGE_SIZE = (640, 640)
feature_extractor = FeatureExtractor()
resize = Resize(IMAGE_SIZE)
resize_112 = Resize((112,112))

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


def calculate_SER_FIQ_score(generated_images_path):
    torch_device = torch.device("cpu")
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    image_list = os.listdir(generated_images_path)
    SER_FIQ_scores = 0
    for image_name in image_list:
      image_path = os.path.join(generated_images_path, image_name)
      resnet = iresnet18(dropout=0.4,num_features=512, use_se=False).to(torch_device)
      resnet.load_state_dict(torch.load("evaluations/checkpoints/resnet18.pth", map_location=torch_device))
      resnet.eval()
      image = io.read_image(image_path)
      image = resize_112(image)
      image = image.type(torch.FloatTensor) / 255.0
      image = normalize(image)
      image = image.unsqueeze(0)
      score = resnet.calculate_serfiq(image, T=10, scaling=5.0)
      SER_FIQ_scores += score.item()
    SER_FIQ_average_score = SER_FIQ_scores/len(image_list)
    return SER_FIQ_average_score

path_to_generated_images = '/content/Portect/assets/swapped_images'
path_to_original_images = '/content/Portect/assets/org_images'

avg_ism, fail_detection_ratio = calculate_FDFR_and_ISM(path_to_generated_images, path_to_original_images)
print(avg_ism, fail_detection_ratio)
SER_FIQ_average_score = calculate_SER_FIQ_score(path_to_generated_images)
print(f"SER-FIQ Score: {SER_FIQ_average_score:.8f}")









