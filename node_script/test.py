#!/usr/bin/env python3
from typing import Optional

import rospy
from node_config import NodeConfig
from rospy import Publisher, Subscriber
from sensor_msgs.msg import Image

from detic_ros.msg import SegmentationInfo


def segmentation_callback(msg: SegmentationInfo):
    image_msg = msg.segmentation
    # TODO: Process the image_msg
    publisher.publish(image_msg)

rospy.init_node('test_node')
subscriber = rospy.Subscriber('/docker/detic_segmentor/segmentation_info', SegmentationInfo, segmentation_callback)
publisher = rospy.Publisher('segmentation', Image, queue_size=10)

rospy.spin()
