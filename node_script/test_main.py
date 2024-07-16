from processor import DeticProcessor
import argparse
import cv2
from detic_config import DeticConfig

import os,glob

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_boxes(image, boxes, names, scores, save_path):
    """
    在图像上绘制每个物体的框，并标注名称和分数。

    :param image: 图像数据（可以是 numpy 数组）
    :param boxes: 物体框列表，每个框为 [x_min, y_min, x_max, y_max]
    :param names: 物体名称列表
    :param scores: 物体分数列表
    """
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, name, score in zip(boxes, names, scores):
        # 解构 box 坐标
        x_min, y_min, x_max, y_max = box

        # 创建矩形框
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # 添加名称和分数标签
        label = f'{name}: {score:.2f}'
        plt.text(x_min, y_min, label, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')  # 隐藏坐标轴
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 关闭图形，释放内存


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detic processor configs")
    parser.add_argument("--model_type", default="swin")
    # parser.add_argument("--dataset", default="/home/lsy/dataset/CoRL_real")
    # parser.add_argument("--video", default="0001")
    parser.add_argument("--output_folder", default="detic_output")
    parser.add_argument("--vocabulary", default="lvis", choices=['lvis', 'custom', 'icra23', 'lvis+icra23',
                                                                 'lvis+ycb_video', 'ycb_video', 'scan_net',
                                                                 'imagenet21k'])
    parser.add_argument("--custom_vocabulary", default="", help="comma separated words")
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--image_dir", default="/home/detic_ws/detic_ros/datasets/isaac_south/color/")
    parser.add_argument("--root_path", default="/home/detic_ws/detic_ros")
    parser.add_argument("--verbose", action='store_true')

    args = parser.parse_args()

    config = DeticConfig(args)
    detic_processor = DeticProcessor(config, args.vocabulary, args.custom_vocabulary)

    img_path_list = glob.glob(os.path.join(args.image_dir, '*.jpg'))
    if not os.path.exists(args.output_folder):
            os.mkdir(args.output_folder)
    for i, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path)
        raw_result = detic_processor.infer(img) # segmentation_raw_image: np.ndarray class_indices: List[int] scores: List[float]pred_boxes: List[int]detected_class_names: List[str]features_np: List[float]
        # seg_img = raw_result.segmentation_raw_image     
        save_path = os.path.join(args.output_folder,f"{i:04d}.jpg")
        plot_boxes(img, raw_result.pred_boxes, raw_result.detected_class_names, raw_result.scores, save_path)
    
    # cv2.imwrite()
    # cv2.waitKey(0)
    