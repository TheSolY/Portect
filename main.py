from utils import FeatureExtractor
from diffusers.utils import load_image

image_path = '/Users/solyarkoni-port-mac/Pictures/sample images/beyonce.webp'

image = load_image(image_path)
feature_extractor = FeatureExtractor()
image_emb = feature_extractor.extract_features(image)
print(image_emb.shape)
