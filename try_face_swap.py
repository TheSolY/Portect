from utils import FaceSwapper

org_img_url = 'https://static01.nyt.com/images/2023/11/14/multimedia/00pol-trump-2025-whatweknow-mlkq/00pol-trump-2025-whatweknow-mlkq-mediumSquareAt3X.jpg'
src_img_url = 'https://www.toolshero.com/wp-content/uploads/2020/07/barack-obama-toolshero.jpg'

face_swapper = FaceSwapper()
face_swapper.swap_face(org_img_url, src_img_url, './face_swap_example.png')
