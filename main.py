from utils import FaceSwapper, farthest_neighbor, single_image_path_from_index
from diffusers.utils import load_image
import torch
import os


image_url = 'https://i.scdn.co/image/ab67616d00001e022aa20611c7fb964a74ab01a6'
# image_url = 'https://www.toolshero.com/wp-content/uploads/2020/07/barack-obama-toolshero.jpg'
# image_url = 'https://static.wikia.nocookie.net/priceisright/images/f/f9/Elizabeth_gutierrez_png_by_mickavolianahi-d4l1olk.png/revision/latest?cb=20170811235544'
celeba_path = '/datasets/celeba'

image = load_image(image_url)
face_swapper = FaceSwapper()
image_emb = face_swapper.extract_features(image)
# print(image_emb.shape)

id_centroids = torch.load('assets/celeba_id_embedding_centroids')
id_identity_idxs = torch.load('assets/unique_ids')
# print(id_centroids.shape)

idx, vec = farthest_neighbor(image_emb, id_centroids.numpy())
identity_idx = id_identity_idxs[idx].item()
print(f'farthest id number: {identity_idx}')

src_id_filename = single_image_path_from_index(identity_idx, os.path.join(celeba_path, 'identity_CelebA.txt'))
srd_id_path = os.path.join(celeba_path, 'img_align_celeba', src_id_filename)

face_swapper.swap_face(image_url, srd_id_path, './1.png')


