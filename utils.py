import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper
from insightface.utils import face_align
from diffusers.utils import load_image
from typing import List, Union
from torch import Tensor


# Manipulation on INSwapper to swap from embedding - just skip the embedding extraction from image step
class EmbSwapper(INSwapper):
    def __init__(self, model_file):
        super(EmbSwapper, self).__init__(model_file=model_file)
        # self.model_file = model_file

    def get(self, img, target_face, source_emb, paste_back=True):
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        latent = source_emb.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
        # print(latent.shape, latent.dtype, pred.shape)
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        if not paste_back:
            return bgr_fake, M
        else:
            target_img = img
            fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            fake_diff[:2, :] = 0
            fake_diff[-2:, :] = 0
            fake_diff[:, :2] = 0
            fake_diff[:, -2:] = 0
            IM = cv2.invertAffineTransform(M)
            img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
            bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            img_white[img_white > 20] = 255
            fthresh = 10
            fake_diff[fake_diff < fthresh] = 0
            fake_diff[fake_diff >= fthresh] = 255
            img_mask = img_white
            mask_h_inds, mask_w_inds = np.where(img_mask == 255)
            mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
            mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h * mask_w))
            k = max(mask_size // 10, 10)
            # k = max(mask_size//20, 6)
            # k = 6
            kernel = np.ones((k, k), np.uint8)
            img_mask = cv2.erode(img_mask, kernel, iterations=1)
            kernel = np.ones((2, 2), np.uint8)
            fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
            k = max(mask_size // 20, 5)
            # k = 3
            # k = 3
            kernel_size = (k, k)
            blur_size = tuple(2 * i + 1 for i in kernel_size)
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
            k = 5
            kernel_size = (k, k)
            blur_size = tuple(2 * i + 1 for i in kernel_size)
            fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
            img_mask /= 255
            fake_diff /= 255
            # img_mask = fake_diff
            img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
            fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
            fake_merged = fake_merged.astype(np.uint8)
            return fake_merged


# functions from InstantID

def convert_image_to_cv2(img: Union[Tensor, Image.Image]) -> np.ndarray:
    if isinstance(img, Tensor):
        img = img.squeeze().detach().numpy().transpose(1, 2, 0)
    else:
        img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


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

    def find_max_face(self, face_img: Union[Tensor, Image.Image]):
        # Finds the largest face in an image
        face_info = self.app.get(convert_image_to_cv2(face_img))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        return face_info

    def extract_features(self, face_image: Union[Tensor, Image.Image]) -> np.ndarray:
        # This function is the phi of the cloak learning model
        face_info = self.find_max_face(face_image)
        face_emb = face_info['embedding']
        return face_emb

    def extract_bounding_box(self, face_image: Union[Tensor, Image.Image]) -> np.ndarray:
        face_info = self.find_max_face(face_image)
        return face_info['bbox']


# end functions from InstantID


# Functions from insightFace face swapper example:
class FaceSwapper(FeatureExtractor):
    def __init__(self, img_size=(640, 640)):
        super(FaceSwapper, self).__init__(img_size=img_size)
        self.swapper = INSwapper('./models/inswapper_128.onnx')

    def swap_face(self, org_image_url: str, src_image_url: str, save_path: str) -> Image:
        # Swaps the face in org_img to that of src_img
        org_img = load_image(org_image_url)
        org_face = self.find_max_face(org_img)

        src_img = load_image(src_image_url)
        src_face = self.find_max_face(src_img)

        res = np.array(org_img.copy())[:, :, ::-1]
        res = self.swapper.get(res, org_face, src_face, paste_back=True)
        cv2.imwrite(save_path, res)


class FaceSwapper2(FeatureExtractor):
    def __init__(self, img_size=(640, 640)):
        super(FaceSwapper2, self).__init__(img_size=img_size)
        self.swapper = EmbSwapper('./models/inswapper_128.onnx')

    def swap_face(self, org_image_url: str, src_emb: np.array, save_path: str) -> Image:
        # Swaps the face in org_img to that of src_emb from embedding
        org_img = load_image(org_image_url)
        org_face = self.find_max_face(org_img)

        res = np.array(org_img.copy())[:, :, ::-1]
        res = self.swapper.get(res, org_face, src_emb, paste_back=True)
        cv2.imwrite(save_path, res)



def row_cosine_similarity(vec1, mat1):
    # row wise cosine similarity between a single embedding vector and a matrix of many embeddings.
    cosine = np.dot(mat1, vec1) / (np.linalg.norm(vec1) * np.linalg.norm(mat1, axis=1) + 1e-7)
    return cosine


def farthest_neighbor(face_emb: np.ndarray, id_centroids: np.ndarray):
    sims = row_cosine_similarity(face_emb, id_centroids)
    idx = np.argmin(sims)       # farthest neighbor is the least similar
    return idx, id_centroids[idx].astype('float32')


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


def all_image_path_from_index(selected_id: int, annot_path: str) -> List[str]:
    # Finds all the image paths that belong to the selected identity
    # consider to do it once and save a lookup table
    selected_id = str(selected_id)
    path_list = []
    with open(annot_path, 'r') as f:
        annot = f.readlines()
        for line in annot:
            filename, id = line.strip().split(' ')
            if id == selected_id:
                path_list.append(filename)
        return path_list


def interpolate_embedding(src_emb: np.ndarray, target_emb: np.ndarray, alpha: float) -> np.ndarray:
    return (1-alpha) * src_emb + alpha * target_emb
