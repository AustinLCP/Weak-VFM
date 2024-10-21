import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取 .npz 文件
htc_mask_data = np.load('PointSAM/PointSAM_val/htc_mask/semantic_2011_09_30_drive_0020_sync_0000000988.png.npz')
sam_semantic_mask_data = np.load('PointSAM/PointSAM_val/CAM_FRONT/semantic_2011_09_30_drive_0020_sync_0000000988.png.npz')
sam_instance_mask_data = np.load('PointSAM/PointSAM_val/CAM_FRONT/instance_2011_09_30_drive_0020_sync_0000000988.png.npz')
image = cv2.imread('data/kitti/raw_data/2011_09_30/2011_09_30_drive_0020_sync/image_02/data/0000000988.png')

# Load the mask using the correct key
sam_semantic_mask = sam_semantic_mask_data['data']
sam_instance_mask = sam_instance_mask_data['data']
htc_mask = htc_mask_data['data']

# Create a figure to display the image and mask
plt.figure(figsize=(10, 10))

# Display the original image
plt.subplot(4, 1, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# Display the mask on top of the image
plt.subplot(4, 1, 2)
plt.imshow(image)
plt.imshow(htc_mask, alpha=0.5)  # Overlay the mask with transparency
plt.title("Image with HTC semantic Mask")
plt.axis('off')

# Display the mask on top of the image
plt.subplot(4, 1, 3)
plt.imshow(image)
plt.imshow(sam_semantic_mask, alpha=0.5)  # Overlay the mask with transparency
plt.title("Image with SAM semantic Mask")
plt.axis('off')

# Display the mask on top of the image
plt.subplot(4, 1, 4)
plt.imshow(image)
plt.imshow(sam_instance_mask, alpha=0.5, cmap='tab20')  # Overlay the mask with transparency
plt.title("Image with SAM instance Mask")
plt.axis('off')

plt.show()

# bin_file = 'data\\kitti\\raw_data\\2011_10_03\\2011_10_03_drive_0027_sync\\velodyne_points\\data\\0000000417.bin'
# txt_file = '0000000417.txt'
# point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
# np.savetxt(txt_file, point_cloud, fmt='%.6f')
