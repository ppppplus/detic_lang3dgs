#!/usr/bin/env python3
from typing import Optional

import rospy
from jsk_recognition_msgs.msg import LabelArray, VectorArray
from node_config import NodeConfig
from rospy import Publisher, Subscriber
from sensor_msgs.msg import Image, CameraInfo
import torch
from wrapper import DeticWrapper
from cv_bridge import CvBridge
from detic_ros.msg import SegmentationInfo, SegmentationInstanceInfo
from mask_rcnn_ros.msg import Result
from detic_ros.srv import DeticSeg, DeticSegRequest, DeticSegResponse
import numpy as np
import cv2 as cv

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
# new add 
import message_filters

from std_msgs.msg import Float32MultiArray
import pcl
from sensor_msgs.point_cloud2 import read_points
import open3d as o3d

import ros_numpy

_cv_bridge = CvBridge()
issac_data = np.genfromtxt('/root/catkin_ws/src/detic_ros/node_script/isaac_sim_config.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
rgb_data = np.genfromtxt('/root/catkin_ws/src/detic_ros/node_script/rgb_map.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
# 通过index检索ID的函数
def get_id_by_index(index):
    ids = issac_data['ID'][issac_data['index'] == index]
    return ids[0] if ids else 0

class DeticRosNode:
    detic_wrapper: DeticWrapper
    sub: Subscriber
    # some debug image publisher
    pub_debug_image: Optional[Publisher]
    pub_debug_segmentation_image: Optional[Publisher]

    # used when you set use_jsk_msgs = True
    pub_segimg: Optional[Publisher]
    pub_labels: Optional[Publisher]
    pub_score: Optional[Publisher]

    # otherwise, the following publisher will be used
    pub_info: Optional[Publisher]
    pub_info_mask: Optional[Publisher]

    # debug 用于检测点云的发布
    sub_pcd: Subscriber

    def __init__(self, node_config: Optional[NodeConfig] = None):
        if node_config is None:
            node_config = NodeConfig.from_rosparam()

        rospy.loginfo("node_config: {}".format(node_config))

        self.detic_wrapper = DeticWrapper(node_config)
        self.srv_handler = rospy.Service('~segment_image', DeticSeg, self.callback_srv)
        self.depth_image = Image() 
        self.underprocessing = False

        if node_config.enable_pubsub:
            # As for large buff_size please see:
            # https://answers.ros.org/question/220502/image-subscriber-lag-despite-queue-1/?answer=220505?answer=220505#post-id-22050://answers.ros.org/question/220502/image-subscriber-lag-despite-queue-1/?answer=220505?answer=220505#post-id-220505
            # self.sub = message_filters.Subscriber('~input_image', Image, self.callback_image, queue_size=1, buff_size=2**24)
            # self.sub_depth = message_filters.Subscriber('~input_depth', Image, self.callback_depth, queue_size=1, buff_size=2**24)
            
            self.sub = message_filters.Subscriber('~input_image', Image)
            self.sub_depth = message_filters.Subscriber('~input_depth', Image)
            self.sub_camera_info = message_filters.Subscriber('~input_camera_info', CameraInfo)


            sync = message_filters.ApproximateTimeSynchronizer([self.sub, self.sub_depth, self.sub_camera_info], 10, 0.1)
            sync.registerCallback(self.callback_image)
            if node_config.use_jsk_msgs:
                self.pub_segimg = rospy.Publisher('~segmentation', Image, queue_size=1)
                self.pub_labels = rospy.Publisher('~detected_classes', LabelArray, queue_size=1)
                self.pub_score = rospy.Publisher('~score', VectorArray, queue_size=1)
            else:
                self.pub_info = rospy.Publisher('~segmentation_info', SegmentationInfo,
                                                queue_size=1)
                self.pub_image = rospy.Publisher('~segmentation_image', Image, queue_size=1)
                self.pub_image_rgb = rospy.Publisher('~segmentation_image_rgb', Image, queue_size=1)
                # origin rgb image & depth image
                self.pub_image_origin_rgb = rospy.Publisher('~segmentation_image_origin_rgb', Image, queue_size=1)
                self.pub_image_origin_depth = rospy.Publisher('~segmentation_image_origin_depth', Image, queue_size=1)
                self.pub_camera_info = rospy.Publisher('~origin_camera_info', CameraInfo, queue_size=1)

                self.pub_info_mask = rospy.Publisher('~segmentation_info_mask', Result, queue_size=1)
                # add by yzc 6_10:新增一个发布实例分割mask的topic
                self.pub_instance_mask = rospy.Publisher('~segmentation_instance_mask', SegmentationInstanceInfo, queue_size=1)
                # add by yzc 6-11 新增发布实例rgb的topic
                self.pub_instance_rgb = rospy.Publisher('~segmentation_instance_rgb', Image, queue_size=1)
            if node_config.out_debug_img:
                self.pub_debug_image = rospy.Publisher('~debug_image', Image, queue_size=1)
            else:
                self.pub_debug_image = None
            if node_config.out_debug_segimg:
                self.pub_debug_segmentation_image = rospy.Publisher('~debug_segmentation_image',
                                                                    Image, queue_size=10)
            else:
                self.pub_debug_segmentation_image = None
            
            self.sub_pcd = rospy.Subscriber('~/camera/depth/points', PointCloud2, self.callback_pcd)
            # self.sub_feature  = rospy.Subscriber('~/docker/your_topic_name', Float32MultiArray, self.callback_feature)

        if node_config.num_torch_thread is not None:
            torch.set_num_threads(node_config.num_torch_thread)

        rospy.loginfo('initialized node')

    def raw_result_to_image(self, raw_result):
        LabelArray = raw_result.get_label_array()
        seg_img = raw_result.get_ros_segmentaion_image()
        # 使用cv_bridge将seg_img转换为maskArray
        mask = _cv_bridge.imgmsg_to_cv2(seg_img, "32SC1")
        print("========mask.shape===============",mask.shape)
        cv.imwrite("/root/catkin_ws/src/detic_ros/node_script/mask.png",mask)
        # 找出其中值不为0的像素点,按照不同的mask进行处理
        for i,label in enumerate(LabelArray.labels):
            isaac_id = get_id_by_index(label.id)
            # print("issac_id",isaac_id)
            mask[mask == i+1] = isaac_id
            # print("OK")
        # 存储mask
        mask = mask.astype(np.int32)
        

        # mask为单通道图像，需要转为三通道图像；通过rgb_data，建立1-20整数到RGB的映射,其格式如下：
        # id,red,green,blue,alpha
        # 0,32,32,32,255
        # 1,255,20,127,255
        rgb_img = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        for i in range(1,21):
            rgb_img[mask == i] = (rgb_data['red'][rgb_data['id'] == i][0],rgb_data['green'][rgb_data['id'] == i][0],rgb_data['blue'][rgb_data['id'] == i][0])
        cv.imwrite("/root/catkin_ws/src/detic_ros/node_script/rgb_img.png",rgb_img)

        
        # 将mask转为rosmsg
        mask_msg_rgb = _cv_bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
        mask_msg_rgb.header = seg_img.header
        mask_msg = _cv_bridge.cv2_to_imgmsg(mask, encoding="32SC1")
        mask_msg.header = seg_img.header
        return mask_msg, mask_msg_rgb

    # 将raw_result转成实例分割的feature和mask的消息，然后发布
    def raw_result_to_instance_mask(self, raw_result):
        # 通过raw_result获取实例分割mask
        segmentation_instance_info = raw_result.get_segmentation_instance_info()
        label_array = raw_result.get_label_array()
        label_array = label_array.labels
        # 用于检查是否对应（结果是对应的）
        # print("label_array",label_array)
        # detected_classes = segmentation_instance_info.detected_classes
        # print("detected_classes",detected_classes)
        mask = _cv_bridge.imgmsg_to_cv2(segmentation_instance_info.segmentation, "32SC1")
        # 将mask转成mask_rgb，其中R通道表示实例的index，G和B通道联合表示实例的instance_id
        mask_rgb = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        for i,label in enumerate(label_array):
            index = i + 1
            instance_id = label.id
            instance_id_G = instance_id // 256
            instance_id_B = instance_id % 256
            # print("index,instance_id_G,instance_id_B",index,instance_id_G,instance_id_B)
            mask_rgb[mask == index] = (index,instance_id_G,instance_id_B)
        mask_rgb_msg = _cv_bridge.cv2_to_imgmsg(mask_rgb, encoding="rgb8")
        segmentation_instance_info.segmentation = mask_rgb_msg
        # 保存mask_rgb 进行debug
        cv.imwrite("/root/catkin_ws/src/detic_ros/node_script/mask_rgb.png",mask_rgb)
        return segmentation_instance_info
    

    # def instance_mask_depth_to_rgb(self, instance_mask):
    #     mask_depth = instance_mask.segmentation
    #     mask_depth = _cv_bridge.imgmsg_to_cv2(mask_depth, "32SC1")
    #     LabelArray = raw_result.get_label_array()
    #     mask_rgb = np.zeros((mask_depth.shape[0],mask_depth.shape[1],3),dtype=np.uint8)



    def callback_image(self, msg: Image, msg_depth: Image, msg_camera_info: CameraInfo):
        # Inference
        current_depth = msg_depth
        current_rgb = msg
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
            seg_info = raw_result.get_segmentation_info()
            self.pub_info.publish(seg_info)
            seg_image, seg_image_rgb = self.raw_result_to_image(raw_result)
            
            self.pub_image.publish(seg_image)

            self.pub_image_rgb.publish(seg_image_rgb)
            # origin rgb image & depth image
            self.pub_image_origin_rgb.publish(current_rgb)
            self.pub_image_origin_depth.publish(current_depth)
            self.pub_camera_info.publish(current_camera_info)

            # add 6-10: 发布带有实例分割instance mask的topic
            instance_mask = self.raw_result_to_instance_mask(raw_result)
            self.pub_instance_mask.publish(instance_mask)

            # add 6-11: 发布实例分割的rgb图像
            instance_rgb = Image()
            instance_rgb = instance_mask.segmentation
            instance_rgb.header = current_rgb.header
            self.pub_instance_rgb.publish(instance_rgb)


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

    def callback_srv(self, req: DeticSegRequest) -> DeticSegResponse:
        msg = req.image
        raw_result = self.detic_wrapper.infer(msg)
        seginfo = raw_result.get_segmentation_info()

        resp = DeticSegResponse()
        resp.seg_info = seginfo

        if raw_result.visualization is not None:
            debug_image = raw_result.get_ros_debug_segmentation_img()
            resp.debug_image = debug_image
        return resp
    
    def callback_feature(self, msg: Float32MultiArray):
        # 从msg中获取feature
        features = msg.data
        # 将feature转为numpy array
        features = np.array(features)
        # 将feature转为二维的numpy array，每行包含一个特征
        features = features.reshape(-1, 512)
        # 将其保存到本地
        np.save("/root/catkin_ws/src/detic_ros/node_script/features.npy",features)
        # print("features.shape",features.shape)

    def callback_pcd(self, msg: PointCloud2):
        # 通过pcl保存点云
        # 从点云中提取三维坐标数值
        # pc = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z","rgb"))
        # pass
        pc = ros_numpy.numpify(msg)
        pc = ros_numpy.point_cloud2.split_rgb_field(pc)
        print("pc.shape",pc.shape)
        heght = pc.shape[0]
        width = pc.shape[1]
        points = np.zeros((heght, width,3))
        points[:,:, 0] = pc['x']
        points[:,:, 1] = pc['y']
        points[:,:, 2] = pc['z']
        rgb = np.zeros((heght, width,3))
        rgb[:,:, 0] = pc['r']
        rgb[:,:, 1] = pc['g']
        rgb[:,:, 2] = pc['b']
        print("points.shape",points.shape)
        print("rgb.shape",rgb.shape)
        # 打印rgb中不为零的值
        # print(rgb)
        # open3d保存点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1,3))
        pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1,3)/255)
        o3d.io.write_point_cloud("/root/catkin_ws/src/detic_ros/node_script/pcd.ply",pcd)
        



if __name__ == '__main__':
    rospy.init_node('detic_node', anonymous=True)
    node = DeticRosNode()
    rospy.spin()
