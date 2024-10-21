import os
import re
import numpy as np
import torch
import json
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from tqdm import tqdm


def groundingDINO_load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def groundingDINO_load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def groundingDINO_get_grounding_output(model, images, caption, text_prompt, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    # assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    # caption = caption.lower()
    # caption = caption.strip()
    # if not caption.endswith("."):
    #     caption = caption + "."
    # device = "cuda" if not cpu_only else "cpu"
    # model = model.to(device)
    # image = image.to(device)
    # with torch.no_grad():
    #     outputs = model(image[None], captions=[caption])
    # logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    # boxes = outputs["pred_boxes"][0]  # (nq, 4)
    #
    # # filter output
    # if token_spans is None:
    #     logits_filt = logits.cpu().clone()
    #     boxes_filt = boxes.cpu().clone()
    #     filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    #     logits_filt = logits_filt[filt_mask]  # num_filt, 256
    #     boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    #
    #     # get phrase
    #     tokenlizer = model.tokenizer
    #     tokenized = tokenlizer(caption)
    #     # build pred
    #     pred_phrases = []
    #     for logit, box in zip(logits_filt, boxes_filt):
    #         pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
    #         if with_logits:
    #             pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
    #         else:
    #             pred_phrases.append(pred_phrase)
    # else:
    #     # given-phrase mode
    #     positive_maps = create_positive_map_from_span(
    #         model.tokenizer(text_prompt),
    #         token_span=token_spans
    #     ).to(image.device) # n_phrase, 256
    #
    #     logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
    #     all_logits = []
    #     all_phrases = []
    #     all_boxes = []
    #     for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
    #         # get phrase
    #         phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
    #         # get mask
    #         filt_mask = logit_phr > box_threshold
    #         # filt box
    #         all_boxes.append(boxes[filt_mask])
    #         # filt logits
    #         all_logits.append(logit_phr[filt_mask])
    #         if with_logits:
    #             logit_phr_num = logit_phr[filt_mask]
    #             all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
    #         else:
    #             all_phrases.extend([phrase for _ in range(len(filt_mask))])
    #     boxes_filt = torch.cat(all_boxes, dim=0).cpu()
    #     pred_phrases = all_phrases
    #
    #
    # return boxes_filt, pred_phrases

    assert text_threshold is not None or token_spans is not None, "text_threshold and token_spans should not be None at the same time!"
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    images = images.to(device)  # images shape: (batch_size, C, H, W)
    batch_size = images.shape[0]

    captions = [caption] * batch_size

    with torch.no_grad():
        outputs = model(images, captions=captions)
    logits = outputs["pred_logits"].sigmoid()  # (batch_size, nq, 256)
    boxes = outputs["pred_boxes"]  # (batch_size, nq, 4)

    pred_boxes_list = []
    pred_phrases_list = []

    for i in range(batch_size):
        logits_i = logits[i]  # (nq, 256)
        boxes_i = boxes[i]  # (nq, 4)

        if token_spans is None:
            logits_filt = logits_i.cpu().clone()
            boxes_filt = boxes_i.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            tokenizer = model.tokenizer
            tokenized = tokenizer(caption)

            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
                if with_logits:
                    pred_phrases.append(f"{pred_phrase}({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)
            pred_boxes_list.append(boxes_filt)
            pred_phrases_list.append(pred_phrases)
        else:
            positive_maps = create_positive_map_from_span(
                model.tokenizer(text_prompt),
                token_span=token_spans
            ).to(images.device)  # n_phrase, 256

            logits_for_phrases = positive_maps @ logits_i.T  # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
                phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                filt_mask = logit_phr > box_threshold
                all_boxes.append(boxes_i[filt_mask])
                all_logits.append(logit_phr[filt_mask])
                if with_logits:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend([f"{phrase}({str(logit.item())[:4]})" for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases
            pred_boxes_list.append(boxes_filt)
            pred_phrases_list.append(pred_phrases)

    return pred_boxes_list, pred_phrases_list


def groundingDINO_pred(batch_size,image_path_list):
    # config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    # checkpoint_path = "groundingdino/weights/groundingdino_swint_ogc.pth"  # change the path of the model
    # image_path_list = os.listdir("data/kitti/KITTI3D/training/image_2")
    # text_prompt = "Car . Pedestrian. Cyclist . Van . Person sitting . Truck . Tram"
    #
    # box_threshold = 0.3
    # text_threshold = 0.25
    # token_spans = None
    # cpu_only = True
    #
    # # load model
    # model = groundingDINO_load_model(config_file, checkpoint_path, cpu_only=cpu_only)
    #
    # # load image
    # pred_results = []
    # for image_path in tqdm(image_path_list, desc="GroundingDINO predict"):
    #     image_full_path = os.path.join("data/kitti/KITTI3D/training/image_2", image_path)
    #     image_pil, image = groundingDINO_load_image(image_full_path)
    #
    #     # set the text_threshold to None if token_spans is set.
    #     if token_spans is not None:
    #         text_threshold = None
    #         print("Using token_spans. Set the text_threshold to None.")
    #
    #     # run model
    #     boxes_filt, pred_phrases = groundingDINO_get_grounding_output(
    #         model, image, text_prompt, text_prompt, box_threshold, text_threshold, cpu_only=cpu_only,
    #         token_spans=eval(f"{token_spans}")
    #     )
    #
    #     # visualize pred
    #     size = image_pil.size
    #     pred_dict = {
    #         "boxes": boxes_filt,
    #         "size": [size[1], size[0]],  # H,W
    #         "labels": pred_phrases,
    #     }
    #
    #     pred_results.append(pred_dict)
    #
    # return pred_results

    config_file = "groundingDINO/config/GroundingDINO_SwinT_OGC.py"  # 模型配置文件路径
    checkpoint_path = "groundingDINO/weights/groundingdino_swint_ogc.pth"  # 模型权重文件路径
    # image_path_list = os.listdir("data/kitti/KITTI3D/training/image_2")
    text_prompt = "Car"

    box_threshold = 0.3
    text_threshold = 0.25
    token_spans = None
    cpu_only = False

    # 加载模型
    model = groundingDINO_load_model(config_file, checkpoint_path, cpu_only=cpu_only)

    pred_results = []

    for i in tqdm(range(0, len(image_path_list), batch_size), desc="GroundingDINO predict"):
        batch_image_paths = image_path_list[i:i + batch_size]
        batch_images_pil = []
        batch_images = []
        for image_path in batch_image_paths:
            image_full_path = os.path.join("data/kitti/KITTI3D/training/image_2", image_path+".png")
            image_pil, image = groundingDINO_load_image(image_full_path)
            batch_images_pil.append(image_pil)
            batch_images.append(image)

        # 获取批次中所有图像的最大高度和宽度
        heights = [img.shape[1] for img in batch_images]
        widths = [img.shape[2] for img in batch_images]
        max_height = max(heights)
        max_width = max(widths)

        # 对每个图像进行填充，使其尺寸与最大高度和宽度一致
        from torchvision.transforms.functional import pad

        for idx in range(len(batch_images)):
            image = batch_images[idx]
            c, h, w = image.shape
            pad_bottom = max_height - h
            pad_right = max_width - w
            padding = (0, 0, pad_right, pad_bottom)  # 左、上、右、下
            image_padded = pad(image, padding, fill=0)
            batch_images[idx] = image_padded

        # 将填充后的图像堆叠成一个批次
        batch_images_tensor = torch.stack(batch_images, dim=0)  # (batch_size, C, H, W)

        if token_spans is not None:
            text_threshold = None
            print("Using token_spans. Set the text_threshold to None.")

        # 运行模型
        boxes_filt_list, pred_phrases_list = groundingDINO_get_grounding_output(
            model, batch_images_tensor, text_prompt, text_prompt, box_threshold, text_threshold,
            cpu_only=cpu_only, token_spans=eval(f"{token_spans}")
        )

        for image_pil, boxes_filt, pred_phrases in zip(batch_images_pil, boxes_filt_list, pred_phrases_list):
            size = image_pil.size
            pred_dict = {
                "boxes": boxes_filt,
                "size": [size[1], size[0]],  # H,W
                "labels": pred_phrases,
            }
            pred_results.append(pred_dict)

    return pred_results


def groundingDINO_pred_result_to_coco_format(kitti_predictions):
    output = []
    categories = [
        {"id": 0, "name": "Car"},
        {"id": 1, "name": "Pedestrian"},
        {"id": 2, "name": "Cyclist"},
        {"id": 3, "name": "Van"},
        {"id": 4, "name": "Person sitting"},
        {"id": 5, "name": "Truck"},
        {"id": 6, "name": "Tram"},
    ]

    # for each image
    image_id = 0
    for result in kitti_predictions:
        for i in range(len(result['labels'])):
            boxes = result['boxes'].tolist()
            phrase = result['labels'][i].split(" ")
            cat_score = phrase[-1]
            match = re.match(r"(\w+)\(([\d.]+)\)", cat_score)
            if match:
                object_class = match.group(1)
                confidence = float(match.group(2))
                category_name = object_class
                score = confidence
            else:
                raise ValueError("输入格式不正确")

            if category_name == "person":
                category_name = "person sitting"

            category_id = next((cat["id"] for cat in categories if cat["name"].lower() == category_name.lower()),
                               None)
            if category_id is None:
                print(category_name)
                raise ValueError("无法识别的类别")

            box = boxes[i]
            H,W = result['size'][0], result['size'][1]
            xmin = float(box[0]) * W
            ymin = float(box[1]) * H
            width_bbox = float(box[2]) * W
            height_bbox = float(box[3]) * H

            coco_format = {
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [xmin, ymin, width_bbox, height_bbox],
                "score": score
            }
            output.append(coco_format)
        image_id += 1

    with open('eval_visual_tools/groundingDINO/dt_groundingDINO.json', 'w') as f:
        json.dump(output, f)



image_path_list = np.loadtxt("data/kitti/data_file/split/train_3d.txt", dtype=str)
results = groundingDINO_pred(16, image_path_list)
groundingDINO_pred_result_to_coco_format(results)







