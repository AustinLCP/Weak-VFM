import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from lang_sam import LangSAM


image_path_list = np.loadtxt("H:\\Honours\\WeakM3D\\WeakM3D\\data\\kitti\\data_file\\split\\train_raw.txt", dtype=str)
# image_path_list = np.loadtxt("train_raw_sample.txt", dtype=str)
model = LangSAM(ckpt_path='H:\\Honours\\LangSAM\\lang-segment-anything\\weights\\sam_vit_h_4b8939.pth')
text_prompt = "car"


for image_path in tqdm(image_path_list, desc='Saving 2d bbox and masks'):

    image_path = image_path.tolist()[0]
    image_path = "H:\\Honours\\WeakM3D\\WeakM3D\\" + image_path

    bbox2d_path =image_path.replace('image_02', 'langSAM_seg_bbox').replace('png', 'txt')
    mask_path = image_path.replace('image_02', 'langSAM_seg_mask').replace('png', 'npz')

    seg_mask_dir = os.path.dirname(bbox2d_path)
    seg_bbox_dir = os.path.dirname(mask_path)
    if not os.path.exists(seg_mask_dir):
        os.makedirs(seg_mask_dir)
    if not os.path.exists(seg_bbox_dir):
        os.makedirs(seg_bbox_dir)

    image_pil = Image.open(image_path).convert("RGB")
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

    if boxes.shape[0] > 0:
        bbox2d = np.hstack([boxes.reshape(-1, 4), logits.reshape(-1, 1)])
        np.savetxt(bbox2d_path, bbox2d, fmt='%f')
        np.savez(mask_path, masks=masks)







