from PIL import Image
from lang_sam import LangSAM
import cv2
import matplotlib.pyplot as plt

model = LangSAM(ckpt_path='weights/sam_vit_h_4b8939.pth')
image_pil = Image.open("./assets/car.jpeg").convert("RGB")
text_prompt = "wheel"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)


image_path = "./assets/car.jpeg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

for bbox in boxes:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    plt.gca().add_patch(plt.Rectangle(
        (bbox[0], bbox[1]), width, height,
        fill=False, edgecolor='red', linewidth=1))

plt.imshow(image)
plt.show()
