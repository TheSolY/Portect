import numpy as np
import torch
from torchvision.datasets import CelebA
from utils import FeatureExtractor
import scipy.io as sio

embedding_size = 512
dataset = CelebA(root='/datasets/', download=False, target_type='identity')
# dataset = CelebA(root='/Users/solyarkoni-port-mac/Datasets', download=False, target_type='identity')
len_dataset = len(dataset)
print(f'nuber of samples: {len_dataset}')

feature_extractor = FeatureExtractor()

features = np.zeros((len_dataset, embedding_size))
ids = np.zeros(len_dataset, dtype=int)
remove_idx = []

for i, (img, lbl) in enumerate(dataset):
    if i % 1000 == 0:
        print(f'Processing {i} / {len_dataset}')
    try:
        features[i] = feature_extractor.extract_features(img)
        ids[i] = lbl
    except Exception:
        remove_idx.append(i)

legit_idxs = list(set(range(len_dataset)) - set(remove_idx))

legit_idxs = torch.LongTensor(legit_idxs)
ftrs = torch.index_select(torch.tensor(features), 0, legit_idxs)
ids = torch.index_select(torch.tensor(ids), 0, legit_idxs)

unique_identities, unique_identities_mapping = torch.unique(ids, sorted=True, return_inverse=True)
print(f'num unique identities: {len(unique_identities)}')


mean_features = torch.zeros((max(unique_identities) + 1, ftrs.size(1)), dtype=torch.float64)
mean_features.scatter_reduce_(0, unique_identities_mapping.view(ids.size(0), 1).expand(-1, ftrs.size(1)), ftrs, reduce="mean", include_self=False)

torch.save(mean_features[:len(unique_identities)], 'assets/celeba_id_embedding_centroids')
torch.save(unique_identities, 'assets/unique_ids')
