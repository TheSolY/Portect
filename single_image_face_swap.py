from utils import FaceSwapper

org_img_url = 'https://i.scdn.co/image/ab67616d00001e022aa20611c7fb964a74ab01a6'
src_img_url = 'https://www.toolshero.com/wp-content/uploads/2020/07/barack-obama-toolshero.jpg'

face_swapper = FaceSwapper()
face_swapper.swap_face(org_img_url, src_img_url, './face_swap_example.png')
