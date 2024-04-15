import numpy as np
import torch
from torchvision.datasets import CelebA
from utils import FeatureExtractor
import scipy.io as sio

embedding_size = 512
dataset = CelebA(root='/Users/solyarkoni-port-mac/Datasets/', download=False, target_type='identity')

print(f'nuber of samples: {len(dataset)}')

feature_extractor = FeatureExtractor()

features = np.zeros((len(dataset), embedding_size))
ids = np.zeros(len(dataset))
legit_idxs = []

for i, (img, lbl) in enumerate(dataset):
    try:
        features[i] = feature_extractor.extract_features(img)
        ids[i] = lbl
        legit_idxs.append(i)
    except:
        print(i)

legit_idxs = torch.LongTensor(legit_idxs)
features = torch.gather(torch.Tensor(features), 0, legit_idxs)
ids = torch.LongTensor(ids, 0, legit_idxs)

unique_identities = torch.unique(ids)
print(f'num unique identities: {unique_identities}')

unique_identities.scatter_reduce(0, ids, features, reduce="mean", include_self=False)

torch.save(features, './celeba_id_embedding_centroids')
