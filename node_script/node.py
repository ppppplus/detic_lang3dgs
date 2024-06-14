#!/usr/bin/env python3
from typing import Optional
import torch
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge

import rospy
from rospy import Publisher, Subscriber
import message_filters

from node_config import NodeConfig
from wrapper import DeticWrapper
from detic_ros.msg import SegmentationInfo, SegmentationInstanceInfo
from jsk_recognition_msgs.msg import LabelArray, VectorArray
from sensor_msgs.msg import Image, CameraInfo


_cv_bridge = CvBridge()

class DeticRosNode:
    detic_wrapper: DeticWrapper

    # subscriber 
    sub_image: Subscriber
    sub_depth: Subscriber
    sub_camera_info: Subscriber
    sub_pcd: Subscriber

    # used when use_jsk_msgs = False
    pub_info: Optional[Publisher]
    pub_image: Optional[Publisher] # hydra_ros_node
    pub_image_rgb: Optional[Publisher]
    pub_image_origin_rgb: Optional[Publisher]
    pub_image_origin_depth: Optional[Publisher] # hydra_ros_node
    pub_camera_info: Optional[Publisher] # hydra_ros_node
    pub_instance_info: Optional[Publisher] # instance_fusion_node

    # used when use_jsk_msgs = True
    pub_segimg: Optional[Publisher]
    pub_labels: Optional[Publisher]
    pub_score: Optional[Publisher]

    # debug publisher
    pub_debug_image: Optional[Publisher]
    pub_debug_segmentation_image: Optional[Publisher]


    def __init__(self, node_config: Optional[NodeConfig] = None):
        if node_config is None:
            node_config = NodeConfig.from_rosparam()

        rospy.loginfo("node_config: {}".format(node_config))

        self.detic_wrapper = DeticWrapper(node_config)

        if node_config.enable_pubsub:
            # topic synchronization
            self.sub_image = message_filters.Subscriber('~input_image', Image)
            self.sub_depth = message_filters.Subscriber('~input_depth', Image)
            self.sub_camera_info = message_filters.Subscriber('~input_camera_info', CameraInfo)
            sync = message_filters.ApproximateTimeSynchronizer([self.sub_image, self.sub_depth, self.sub_camera_info], 10, 0.1)
            sync.registerCallback(self.callback_image)

            if node_config.use_jsk_msgs:
                self.pub_segimg = rospy.Publisher('~seg_image', Image, queue_size=1)
                self.pub_labels = rospy.Publisher('~detected_classes', LabelArray, queue_size=1)
                self.pub_score = rospy.Publisher('~score', VectorArray, queue_size=1)
            else:
                self.pub_info = rospy.Publisher('~seg_info', SegmentationInfo, queue_size=1)
                self.pub_instance_info = rospy.Publisher('~seg_instance_info', SegmentationInstanceInfo, queue_size=1)
                self.pub_image = rospy.Publisher('~seg_image', Image, queue_size=1)
                self.pub_image_rgb = rospy.Publisher('~seg_image_rgb', Image, queue_size=1)
                
                # origin rgb image & depth image
                self.pub_image_origin_rgb = rospy.Publisher('~origin_rgb', Image, queue_size=1)
                self.pub_image_origin_depth = rospy.Publisher('~origin_depth', Image, queue_size=1)
                self.pub_camera_info = rospy.Publisher('~origin_camera_info', CameraInfo, queue_size=1)

            
            if node_config.out_debug_img:
                self.pub_debug_image = rospy.Publisher('~debug_image', Image, queue_size=1)
            else:
                self.pub_debug_image = None
            if node_config.out_debug_segimg:
                self.pub_debug_segmentation_image = rospy.Publisher('~debug_segmentation_image', Image, queue_size=10)
            else:
                self.pub_debug_segmentation_image = None

        if node_config.num_torch_thread is not None:
            torch.set_num_threads(node_config.num_torch_thread)

        rospy.loginfo('initialized node')

    def callback_image(self, msg: Image, msg_depth: Image, msg_camera_info: CameraInfo):
        # Inference
        current_rgb = msg
        current_depth = msg_depth
        current_camera_info = msg_camera_info
        raw_result = self.detic_wrapper.infer(msg)

        # Publish main topics
        if self.detic_wrapper.node_config.use_jsk_msgs:
            # assertion for mypy
            assert self.pub_segimg is not None
            assert self.pub_labels is not None
            assert self.pub_score is not None
            seg_img = raw_result.get_ros_segmentaion_image()
            labels = raw_result.get_label_array()
            scores = raw_result.get_score_array()
            
            self.pub_segimg.publish(seg_img)
            self.pub_labels.publish(labels)
            self.pub_score.publish(scores)
        else:
            assert self.pub_info is not None
            # seg_info = raw_result.get_segmentation_info()
            instance_info = raw_result.get_segmentation_instance_info()
            # self.pub_info.publish(seg_info)
            self.pub_instance_info.publish(instance_info)

            seg_image = raw_result.get_ros_segmentaion_image()
            seg_image_rgb = raw_result.get_ros_segmentaion_image_rgb()
            self.pub_image.publish(seg_image)
            self.pub_image_rgb.publish(seg_image_rgb)

            # Original rgb image & depth image & camera info
            self.pub_image_origin_rgb.publish(current_rgb)
            self.pub_image_origin_depth.publish(current_depth)
            self.pub_camera_info.publish(current_camera_info)

        # Publish optional topics
        if self.pub_debug_image is not None:
            debug_img = raw_result.get_ros_debug_image()
            self.pub_debug_image.publish(debug_img)

        if self.pub_debug_segmentation_image is not None:
            debug_seg_img = raw_result.get_ros_debug_segmentation_img()
            self.pub_debug_segmentation_image.publish(debug_seg_img)

        # Print debug info
        if self.detic_wrapper.node_config.verbose:
            time_elapsed_total = (rospy.Time.now() - msg.header.stamp).to_sec()
            rospy.loginfo('total elapsed time in callback {}'.format(time_elapsed_total))

if __name__ == '__main__':
    rospy.init_node('detic_node', anonymous=True)
    node = DeticRosNode()
    rospy.spin()
