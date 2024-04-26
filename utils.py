import cv2
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper
from insightface.data import get_image as ins_get_image
from diffusers.utils import load_image
import torch
from diffusers.models import ControlNetModel


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
        self.img_size = img_size

        self.app = FaceAnalysis(name='buffalo_l', root='./', providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=img_size)

    def find_max_face(self, face_img: Image):
        # Finds the largest face in an image
        face_info = self.app.get(cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        return face_info

    def extract_features(self, face_image: Image) -> np.ndarray:
        # This function is the phi of the cloak learning model
        face_info = self.find_max_face(face_image)
        # face_image = resize_img(face_image)
        # face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        # face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        face_emb = face_info['embedding']
        return face_emb

# end functions from InstantID


# Functions from insightFace face swapper example:
class FaceSwapper(FeatureExtractor):
    def __init__(self, img_size=(640, 640)):
        super(FaceSwapper, self).__init__(img_size=img_size)
        self.swapper = INSwapper('./models/inswapper_128.onnx')

    def swap_face(self, org_image_url: str, src_image_url: str, save_path: str) -> Image:
        # Swaps the face in org_img to that of src_img
        # we should replace later the url with local path
        org_img = load_image(org_image_url)
        org_face = self.find_max_face(org_img)

        src_img = load_image(src_image_url)
        src_face = self.find_max_face(src_img)

        res = np.array(org_img.copy())[:, :, ::-1]
        res = self.swapper.get(res, org_face, src_face, paste_back=True)
        cv2.imwrite(save_path, res)


def row_cosine_similarity(vec1, mat1):
    # row wise cosine similarity between a single embedding vector and a matrix of many embeddings.
    cosine = np.dot(mat1, vec1) / (np.linalg.norm(vec1) * np.linalg.norm(mat1, axis=1) + 1e-7)
    return cosine


def farthest_neighbor(face_emb: np.ndarray, id_centroids: np.ndarray):
    sims = row_cosine_similarity(face_emb, id_centroids)
    idx = np.argmin(sims)       # farthest neighbor is the least similar
    return idx, id_centroids[idx]


def single_image_path_from_index(selected_id: int, annot_path: str) -> str:
    # Finds an image that belongs to the selected identity and returns its path
    selected_id = str(selected_id)
    with open(annot_path, 'r') as f:
        annot = f.readlines()
        for line in annot:
            filename, id = line.strip().split(' ')
            if id == selected_id:
                return filename
    raise FileNotFoundError