import os
import sys
from dataclasses import dataclass
from typing import Optional

# import rospkg
# import rospy
import torch

# Dirty but no way, because CenterNet2 is not package oriented
sys.path.insert(0, os.path.join(sys.path[0], 'third_party/CenterNet2/'))

from centernet.config import add_centernet_config
from detectron2.config import get_cfg
from detic.config import add_detic_config


class DeticConfig:
    def __init__(self, args):
        self.model_type = args.model_type
        self.output_folder = args.output_folder
        self.vocabulary = args.vocabulary
        self.custom_vocabulary = args.custom_vocabulary
        self.confidence_threshold = args.confidence_threshold
        # self.image_path = args.image_path
        self.root_path = args.root_path
        self.verbose = args.verbose
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"

        model_names = {
            'swin': 'Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size',
            'convnet': 'Detic_LCOCOI21k_CLIP_CXT21k_640b32_4x_ft4x_max-size',
            'res50': 'Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size',
            'res18': 'Detic_LCOCOI21k_CLIP_R18_640b32_4x_ft4x_max-size',
        }
        assert self.model_type in model_names.keys(), "unsupported model type!"
        self.model_name = model_names[self.model_type]

        self.detic_config_path = os.path.join(
            self.root_path, 'detic_configs',
            self.model_name + '.yaml')

        self.model_weights_path = os.path.join(
            self.root_path, 'models',
            self.model_name + '.pth')

    def to_detectron_config(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = self.device_name
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(self.detic_config_path)
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.confidence_threshold
        cfg.merge_from_list(['MODEL.WEIGHTS', self.model_weights_path])

        # Similar to https://github.com/facebookresearch/Detic/demo.py
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

        # Maybe should edit detic_configs/Base-C2_L_R5021k_640b64_4x.yaml
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = os.path.join(
            self.root_path, 'datasets/metadata/lvis_v1_train_cat_info.json')
        print(cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH)
        cfg.freeze()
        return cfg
