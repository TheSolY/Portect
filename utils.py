import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
import torch
from typing import Tuple


# functions from InstantID

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


class FeatureExtractor:
    def __init__(self, img_size=(640, 640)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        if self.device == torch.device('cpu'):
            self.app = FaceAnalysis(name='buffalo_l', root='./', providers=['CPUExecutionProvider'])
        else:
            self.app = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=img_size)

    def extract_features(self, face_image: Image) -> np.ndarray:
        # This function is the phi of the cloak learning model
        face_image = resize_img(face_image)
        face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        face_emb = face_info['embedding']
        return face_emb

# end functions from InstantID


def row_cosine_similarity(vec1, mat1):
    # row wise cosine similarity between a single embedding vector and a matrix of many embeddings.
    cosine = np.dot(mat1, vec1) / (np.linalg.norm(vec1) * np.linalg.norm(mat1, axis=1) + 1e-7)
    return cosine


def farthest_neighbor(face_emb: np.ndarray, id_centroids: np.ndarray):
    sims = row_cosine_similarity(face_emb, id_centroids)
    idx = np.argmin(np.abs(sims))       # farthest neighbor is the least similar
    return idx, id_centroids[idx]
