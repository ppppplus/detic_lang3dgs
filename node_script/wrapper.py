import os
from dataclasses import dataclass
from typing import List, Optional
import argparse
import pickle

import detic
import numpy as np
import rospkg
import rospy
import torch
from cv_bridge import CvBridge
from detectron2.utils.visualizer import VisImage
from detectron2.config import get_cfg
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo
from node_config import NodeConfig

from jsk_recognition_msgs.msg import Label, LabelArray, VectorArray
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Float32MultiArray
from detic_ros.msg import SegmentationInfo, SegmentationInstanceInfo


_cv_bridge = CvBridge()
issac_data = np.genfromtxt('/root/catkin_ws/src/detic_ros/node_script/configs/isaac_sim_config.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
rgb_data = np.genfromtxt('/root/catkin_ws/src/detic_ros/node_script/configs/rgb_map.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')

def get_id_by_index(index):
    ids = issac_data['ID'][issac_data['index'] == index]
    return ids[0] if ids else 0
 

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--config-file", default="/root/catkin_ws/src/detic_ros/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    parser.add_argument("--dataset", default="/home/lsy/dataset/CoRL_real")
    parser.add_argument("--video", default="0001")
    parser.add_argument("--output_folder", default="detic_output")
    parser.add_argument("--vocabulary", default="imagenet21k", choices=['lvis', 'custom', 'icra23', 'lvis+icra23',
                                                                 'lvis+ycb_video', 'ycb_video', 'scan_net',
                                                                 'imagenet21k'])
    parser.add_argument("--custom_vocabulary", default="", help="comma separated words")
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("~input_image", default="")
    parser.add_argument("~input_depth", default="")
    parser.add_argument("~input_camera_info", default="")
    parser.add_argument("__name", default="detic_segmentor")
    parser.add_argument("__log", default="/root/.ros/log/20e5783e-1907-11ef-82aa-00d49ebf3ea2/docker-detic_segmentor-2.log")
    parser.add_argument("--opts", help="'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)
    # --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    # parser.add_argument("--opts", default=['MODEL.WEIGHTS', '/root/catkin_ws/src/detic_ros/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'], nargs=argparse.REMAINDER)
    return parser

def setup_cfg(args):
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.WEIGHTS = '/root/catkin_ws/src/detic_ros/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
    cfg.DATALOADER.NUM_WORKERS = 2
    
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


@dataclass(frozen=True)
class InferenceRawResult:
    segmentation_raw_image: np.ndarray
    class_indices: List[int]
    scores: List[float]
    visualization: Optional[VisImage]
    header: Header
    detected_class_names: List[str]
    features: Optional[Float32MultiArray] = None

    def get_ros_segmentaion_image(self) -> Image:
        seg_img = _cv_bridge.cv2_to_imgmsg(self.segmentation_raw_image, encoding="32SC1")
        seg_img.header = self.header
        return seg_img
    
    def get_ros_segmentaion_image_rgb(self) -> Image:
        LabelArray = self.get_label_array()
        seg_img = self.get_ros_segmentaion_image()
        mask = _cv_bridge.imgmsg_to_cv2(seg_img, "32SC1")
        for i,label in enumerate(LabelArray.labels):
            isaac_id = get_id_by_index(label.id)
            mask[mask == i+1] = isaac_id
        mask = mask.astype(np.int32)
 
        rgb_img = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        for i in range(1,21):
            rgb_img[mask == i] = (rgb_data['red'][rgb_data['id'] == i][0],rgb_data['green'][rgb_data['id'] == i][0],rgb_data['blue'][rgb_data['id'] == i][0])
        
        seg_img_rgb = _cv_bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
        seg_img_rgb.header = seg_img.header
        return seg_img_rgb

    def get_ros_debug_image(self) -> Image:
        message = "you didn't configure the wrapper so that it computes the debug images"
        assert self.visualization is not None, message
        debug_img = _cv_bridge.cv2_to_imgmsg(
            self.visualization.get_image(), encoding="rgb8")
        debug_img.header = self.header
        return debug_img

    def get_ros_debug_segmentation_img(self) -> Image:
        human_friendly_scaling = 255 // self.segmentation_raw_image.max()
        new_data = (self.segmentation_raw_image * human_friendly_scaling).astype(np.uint8)
        debug_seg_img = _cv_bridge.cv2_to_imgmsg(new_data, encoding="mono8")
        debug_seg_img.header = self.header
        return debug_seg_img

    def get_label_array(self) -> LabelArray:
        labels = [Label(id=i + 1, name=name)
                  for i, name
                  in zip(self.class_indices, self.detected_class_names)]
        lab_arr = LabelArray(header=self.header, labels=labels)
        return lab_arr

    def get_score_array(self) -> VectorArray:
        vec_arr = VectorArray(header=self.header, vector_dim=len(self.scores), data=self.scores)
        return vec_arr

    def get_segmentation_info(self) -> SegmentationInfo:
        seg_img = self.get_ros_segmentaion_image()
        seg_info = SegmentationInfo(detected_classes=self.detected_class_names,
                                    scores=self.scores,
                                    segmentation=seg_img,
                                    header=self.header)
        return seg_info

    def get_segmentation_instance_info(self) -> SegmentationInstanceInfo:
        seg_img = self.get_ros_segmentaion_image()
        seg_instance_info = SegmentationInstanceInfo(detected_classes=self.detected_class_names,
                                                     scores=self.scores,
                                                     segmentation=seg_img,
                                                     features=self.features,
                                                     header=self.header)
        return seg_instance_info


class DeticWrapper:
    predictor: VisualizationDemo
    node_config: NodeConfig
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

    def __init__(self, node_config: NodeConfig):
        self._adhoc_hack_metadata_path()
        detectron_cfg = node_config.to_detectron_config()

        dummy_args = self.DummyArgs(node_config.vocabulary, node_config.custom_vocabulary)
        self.predictor = VisualizationDemo(detectron_cfg, dummy_args)
        args = get_parser().parse_args()
        cfg = setup_cfg(args)
        self.node_config = node_config
        self.class_names = self.predictor.metadata.get("thing_classes", None)
        # print(self.class_names)
        print("class name type ", np.shape(self.class_names))
        data_array = np.array(self.class_names)
        np.savetxt('output.txt', data_array, fmt='%s')
        print("save into txt file")
         

    @staticmethod
    def _adhoc_hack_metadata_path():
        # because original BUILDIN_CLASSIFIER is somehow position dependent
        rospack = rospkg.RosPack()
        pack_path = rospack.get_path('detic_ros')
        path_dict = detic.predictor.BUILDIN_CLASSIFIER
        for key in path_dict.keys():
            path_dict[key] = os.path.join(pack_path, path_dict[key])

    def infer(self, msg: Image) -> InferenceRawResult:
        # Segmentation image, detected classes, detection scores, visualization image
        img = _cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.node_config.verbose:
            time_start = rospy.Time.now()

        if self.node_config.out_debug_img:
            predictions, visualized_output = self.predictor.run_on_image(img)
        else:
            predictions = self.predictor.predictor(img)
            visualized_output = None
        instances = predictions['instances'].to(torch.device("cpu"))
        instances = self.predictor.predict_instances_only(img)
        # print(type(instances))
        if self.node_config.verbose:
            time_elapsed = (rospy.Time.now() - time_start).to_sec()
            rospy.loginfo('elapsed time to inference {}'.format(time_elapsed))
            rospy.loginfo('detected {} classes'.format(len(instances)))

        pred_masks = list(instances.pred_masks)
        scores = instances.scores.tolist()
        class_indices = instances.pred_classes.tolist()
        features = instances.pred_box_features
        features_np = [feature.numpy().astype(np.float32) for feature in features]

        features_msg = Float32MultiArray()
        features_msg.data = np.concatenate(features_np).flatten() # Flatten the array to 1D

        if len(scores) > 0 and self.node_config.output_highest:
            best_index = np.argmax(scores)
            pred_masks = [pred_masks[best_index]]
            scores = [scores[best_index]]
            class_indices = [class_indices[best_index]]

            if self.node_config.verbose:
                rospy.loginfo("{} with highest score {}".format(self.class_names[class_indices[0]], scores[best_index]))

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
            visualized_output,
            msg.header,
            detected_classes_names,
            features_msg)
        return result
