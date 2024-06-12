import os
import sys
import argparse
import pickle
import requests
from pathlib import Path
from time import perf_counter

from PIL import Image
# from transformers import AutoProcessor, AlignModel
from transformers import CLIPProcessor, CLIPModel

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="~/dataset/ScanNet/aligned_scans")
    parser.add_argument("--detic_exp", type=str, default="imagenet21k-0.3")
    parser.add_argument("--video", default="scene0645_00")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stride", type=int, default=3, help="stride of frames to process")
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser()

    # ALIGN model
    # model = AlignModel.from_pretrained("kakaobrain/align-base")
    # processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

    # CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    color_im_folder = dataset / args.video / "color"
    detic_output_folder = dataset / args.video / "detic_output" / args.detic_exp / "instances"
    align_output_folder = dataset / args.video / "detic_align_output" / args.detic_exp / "instances"

    color_im_paths = sorted(color_im_folder.iterdir())

    filtered_color_im_paths = []
    for i in range(0, len(color_im_paths), args.stride):
        filtered_color_im_paths.append(color_im_paths[i])

    for color_im_path in tqdm(filtered_color_im_paths, disable=False):
        output_path = align_output_folder / color_im_path.name.replace(".jpg", ".pkl")
        if output_path.exists():
            continue

        image = cv2.imread(str(color_im_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detic_output_path = detic_output_folder / color_im_path.name.replace("jpg", "pkl")
        with open(detic_output_path, 'rb') as fp:
            instances = pickle.load(fp)

        boxes = instances.get("pred_boxes").tensor.to(args.device)  # XYXY
        if boxes.shape[0] == 0:
            instances.align_feature = []
        else:
            cropped_imgs = []
            for box in boxes:
                cropped_img = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                cropped_img = Image.fromarray(np.copy(cropped_img))
                cropped_imgs.append(cropped_img)
            inputs = processor(images=cropped_imgs, return_tensors="pt")
            box_features = model.get_image_features(**inputs).detach().cpu()
            instances.align_features = box_features

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(instances, f)

        # draw output image
        # plt.figure(figsize=(10, 10))
        # for box, mask, quality in zip(boxes, masks, qualities):
        #     print(f"{quality = }")
        #     plt.imshow(image)
        #     show_mask(mask, plt.gca(), random_color=True)
        #     show_box(box.cpu().numpy(), plt.gca(), f"{quality = }")
        #     plt.show()
        # break


if __name__ == "__main__":
    main()


# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# tic = perf_counter()
# inputs = processor(images=[image] * 20, return_tensors="pt")
# image_features = model.get_image_features(**inputs)
# print(f"{image_features.shape = }")
# print(perf_counter() - tic)

# processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
# model = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined")
#
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# print(f"{image.size = }")
# image.show()
#
# tic = perf_counter()
# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# print(f"{last_hidden_state.shape = }")
# print(perf_counter() - tic)
