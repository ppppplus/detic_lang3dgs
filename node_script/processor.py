import os
from dataclasses import dataclass
from typing import List, Optional
import argparse
import pickle

import detic
import numpy as np
# import rospkg
# import rospy
import torch
from detectron2.utils.visualizer import VisImage
from detectron2.config import get_cfg
import sys
import time
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo
# from node_config import NodeConfig
# from detectron2.config import get_cfg
from detic_config import DeticConfig


@dataclass(frozen=True)
class InferenceRawResult:
    segmentation_raw_image: np.ndarray
    class_indices: List[int]
    scores: List[float]
    pred_boxes: List[int]
    detected_class_names: List[str]
    features_np: List[float]
    
class DeticProcessor:
    predictor: VisualizationDemo
    detic_config: DeticConfig
    class_names: List[str]

    class DummyArgs:
        vocabulary: str

        def __init__(self, vocabulary, custom_vocabulary):
            assert vocabulary in ['lvis', 'openimages', 'objects365', 'coco', 'custom']
            self.vocabulary = vocabulary
            self.custom_vocabulary = custom_vocabulary

    class testArgs:
        vocabulary = 'imagenet21k'
        custom_vocabulary = ''
        pred_all_class = False
        confidence_threshold = 0.3
        save_vis = False
        opts = ['MODEL.WEIGHTS', 'src/detic_ros/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth']
        config_file = 'src/detic_ros/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'


    def __init__(self, detic_config, vocabulary, custom_vocabulary):
        detectron_cfg = detic_config.to_detectron_config()

        dummy_args = self.DummyArgs(vocabulary, custom_vocabulary)
        self.predictor = VisualizationDemo(detectron_cfg, dummy_args)
        self.detic_config = detic_config
        self.class_names = self.predictor.metadata.get("thing_classes", None)
        print(self.class_names)
        print("class name type ", np.shape(self.class_names))
        # data_array = np.array(self.class_names)
        # np.savetxt('output.txt', data_array, fmt='%s')
        # print("save into txt file")     

    def infer(self, image) -> InferenceRawResult:
        # Segmentation image, detected classes, detection scores, visualization image
        img = image.copy()

        if self.detic_config.verbose:
            time_start = time.time()

        # if self.node_config.out_debug_img:
        #     predictions, visualized_output = self.predictor.run_on_image(img)
        # else:
        # predictions = self.predictor.predictor(img)
        # visualized_output = None
        
        # instances = predictions['instances'].to(torch.device("cpu"))
        instances = self.predictor.predict_instances_only(img)
        # print(type(instances))
        if self.detic_config.verbose:
            time_elapsed = (time.time() - time_start).to_sec()
            # rospy.loginfo('elapsed time to inference {}'.format(time_elapsed))
            print('detected {} classes'.format(len(instances)))

        boxes = instances.pred_boxes
        pred_boxes = [box.numpy().astype(np.uint32) for box in boxes]
        pred_masks = list(instances.pred_masks)
        scores = instances.scores.tolist()
        class_indices = instances.pred_classes.tolist()
        features = instances.pred_box_features
        features_np = [feature.numpy().astype(np.float32) for feature in features]

        # features_msg = Float32MultiArray()
        # if len(features_np) > 0:
        #     features_msg.data = np.concatenate(features_np).flatten() # Flatten the array to 1D

        # Initialize segmentation data
        data = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)

        # largest to smallest order to reduce occlusion.
        sorted_index = np.argsort([-mask.sum() for mask in pred_masks])
        for i in sorted_index:
            mask = pred_masks[i]
            # label 0 is reserved for background label, so starting from 1
            data[mask] = (i + 1)

        # Get class and score arrays
        detected_classes_names = [self.class_names[i] for i in class_indices]

        result = InferenceRawResult(
            data,
            class_indices,
            scores,
            pred_boxes,
            detected_classes_names,
            features_np)
        return result
