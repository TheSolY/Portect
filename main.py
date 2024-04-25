from utils import FeatureExtractor, farthest_neighbor
from diffusers.utils import load_image
import torch


# image_url = 'https://resize-elle.ladmedia.fr/r/300,,forcex/crop/300,386,center-middle,forcex,ffffff/img/var/plain_site/storage/images/personnalites/taylor-swift/42391722-2-fre-FR/Taylor-Swift.jpg'
image_url = 'https://static.wikia.nocookie.net/priceisright/images/f/f9/Elizabeth_gutierrez_png_by_mickavolianahi-d4l1olk.png/revision/latest?cb=20170811235544'

image = load_image(image_url)
feature_extractor = FeatureExtractor()
image_emb = feature_extractor.extract_features(image)
print(image_emb.shape)

id_centroids = torch.load('assets/celeba_id_embedding_centroids')
id_identity_idxs = torch.load('assets/unique_ids')
print(id_centroids.shape)

idx, vec = farthest_neighbor(image_emb, id_centroids.numpy())

print(id_identity_idxs[idx])
