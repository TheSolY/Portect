import argparse
import os
from utils import FaceSwapper2, farthest_neighbor, interpolate_embedding, resize_img
from diffusers.utils import load_image
import numpy as np
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cloak your images before using them online.'
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory where the images to cloak are located."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./outputs',
        help="Directory where the cloak images are saved."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Interpolation factor for the target image"
    )
    parser.add_argument(
        "--id_centroids_path",
        type=str,
        default='assets/celeba_id_embedding_centroids',
        help="Path to the target id embeddings file."
    )
    args = parser.parse_args()
    return args


def main(args):
    image_dir = args.image_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f'{image_dir} not found')

    face_swapper = FaceSwapper2()

    # Find the embedding of the images to cloak
    org_images_emb = []
    for i, filename in enumerate(os.listdir(image_dir)):
        _, file_extention = os.path.splitext(filename)
        if file_extention in ('.jpg', '.jpeg', '.png', '.JPG'):
            image = load_image(os.path.join(image_dir, filename))
            emb = face_swapper.extract_features(image)
            org_images_emb.append(emb)

    # Calculate the cloak
    org_images_emb = np.stack(org_images_emb)
    org_images_emb_centroid = org_images_emb.mean(axis=0)
    id_centroids = torch.load(args.id_centroids_path)
    _, vec = farthest_neighbor(org_images_emb_centroid, id_centroids.numpy())
    target_emb = interpolate_embedding(org_images_emb, vec, 0.05)

    # Cloak and save the cloaked images
    for i, image_filename in enumerate(os.listdir(image_dir)):
        _, file_extention = os.path.splitext(image_filename)
        if file_extention in ('.jpg', '.jpeg', '.png', '.JPG'):
            face_swapper.swap_face(os.path.join(image_dir, image_filename),
                                   target_emb[i],
                                   os.path.join(args.output_dir, image_filename))


if __name__ == "__main__":
    args = parse_args()
    main(args)

