import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('data\\kitti\\raw_data\\2011_09_30\\2011_09_30_drive_0020_sync\\image_02\\data\\0000000988.png')
langSAM_mask_npz = np.load("data\\kitti\\raw_data\\2011_09_30\\2011_09_30_drive_0020_sync\\YOLO_seg_mask\\data\\0000000988.npz")
weakM3D_mask_npz = np.load("data\\kitti\\raw_data\\2011_09_30\\2011_09_30_drive_0020_sync\\seg_mask\\data\\0000000988.npz")

langSAM_mask = langSAM_mask_npz['masks']
weakM3D_mask = weakM3D_mask_npz['masks']

langSAM_mask = np.any(langSAM_mask, axis=0)
weakM3D_mask = np.max(weakM3D_mask, axis=0)


plt.figure(figsize=(10, 10))

# Display the original image
plt.subplot(3, 1, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')


plt.subplot(3, 1, 2)
# plt.imshow(image)
plt.imshow(langSAM_mask, alpha=0.5)  # Overlay the mask with transparency
plt.title("YOLO-World Mask")
plt.axis('off')

plt.subplot(3, 1, 3)
# plt.imshow(image)
plt.imshow(weakM3D_mask, alpha=0.5)  # Overlay the mask with transparency
plt.title("WeakM3D Mask")
plt.axis('off')

plt.show()
