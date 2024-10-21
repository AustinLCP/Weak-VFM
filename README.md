# Weakly-Supervised Monocular 3D Object Detection Empowered by Visual Foundation Models in Autonomous Driving

![Overvew](README_Resource/Framework.drawio.png)

## Abstract
Currently, many methods for the 3d bounding box prediction task in autonomous driving require large amounts of annotated 3D data, which is both time-consuming and expensive to obtain. 
The WeakM3D uses 2d bounding boxes and Region of Interest (RoI) LiDAR points as weakly-supervised learning label to alleviate this problem. 
Based on this, I leveraged several Visual Foundation Models to empower the label generation with less human labor. 
Specifically, I introduced the **Segment-Anything Model (SAM)** where object segmentation can be performed on arbitrary input images, to achieve "zero-shot" label generation, 
which released the need for additional pre-training on specific categories to generate labels. 
Besides, I introduced a text-prompting schema containing **Contrastive Language–Image Pretraining (CLIP)** to realize “open-vocabulary” label generation, 
allowing users to use text to manipulate label generation with ideal classes without pre-training. 
Finally, I introduced **Distillation with No Labels (DINO)** and integrated it with SAM and text-prompting together to produce high-quality labels without pre-training and any hand-annotated labels. 
All three label generation methods mentioned above can generate high-quality labels and trained by these labels, the detector showed performance improvement on the KITTI dataset.



## Result
| Method                        | ap 40 bev 0.5        |                      |                      | ap 40 3d 0.5        |                      |                      |
|-------------------------------|----------------------|----------------------|----------------------|---------------------|---------------------|---------------------|
|                               | Easy                 | Mod                  | Hard                 | Easy                | Mod                 | Hard                |
| WeakM3D (Official)            | 60.72                | 40.32                | 31.34                | 53.28               | 33.3                | 25.76               |
| WeakM3D (Reproduction)        | 54.51                | 35.81                | 28.29                | 46.76               | 29.7                | 22.95               |
| PointSAM                      | 52.45                | 36.19                | 28.98                | 46.44               | 30.48               | 23.96               |
| YOLO-World                    | 58.57                | 38.2                 | 30.51                | 52.02               | 32.34               | 25.24               |
| LangSAM                       |                      |                      |                      |                     |                     |                     |

*Table 1. The performance of the detector evaluated in AP40 with IoU=0.5*


| Method                        | ap 11 bev 0.7        |                      |                      | ap 11 3d 0.7        |                      |                      |
|-------------------------------|----------------------|----------------------|----------------------|---------------------|---------------------|---------------------|
|                               | Easy                 | Mod                  | Hard                 | Easy                | Mod                 | Hard                |
| WeakM3D (Official)            | 26.92                | 18.57                | 15.86                | 18.27               | 12.95               | 11.5                |
| WeakM3D (Reproduction)        | 15.69                | 11.8                 | 10.24                | 10.2                | 7.89                | 7.34                |
| PointSAM                      | 22.55                | 15.93                | 14.78                | 17.08               | 12.98               | 11.58               |
| YOLO-World                    | 22.73                | 16.01                | 14.45                | 15.86               | 11.48               | 11.06               |
| LangSAM                       |                      |                      |                      |                     |                     |                     |

*Table 2. The performance of the detector evaluated in AP11 with IoU=0.7*



## Data Preparation
I used KITTI as dataset, all the preparation steps please refer to the *Dataset Preparation* section of [WeakM3D](https://github.com/SPengLiang/WeakM3D)



## Implementation details
python=3.8

Pytorch= 2.2

CUDA=11.8 



## Acknowledgement
The code benefits from:
+ [WeakM3D](https://github.com/SPengLiang/WeakM3D)
+ [PointSAM](https://github.com/BraveGroup/PointSAM-for-MixSup)
+ [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
+ [LangSAM](https://github.com/luca-medeiros/lang-segment-anything)





