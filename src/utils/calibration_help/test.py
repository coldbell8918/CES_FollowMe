#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from sensor_msgs.msg import CameraInfo
from leg_tracker.msg import  PersonArray
from human_following.msg import camera_persons
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf
from geometry_msgs.msg import PoseStamped

class Calibration(object):
    def __init__(self):
        # self.subs2=[rospy.Subscriber('people_tracked', PersonArray, self.lidar_callback)]
        self.subs3=[rospy.Subscriber('tracker/data', camera_persons, self.camera_callback)]
        self.tf = tf.TransformListener()
        self.tt = tf.Transformer()
        
    def camera_callback(self,data):
        if self.tf.frameExists("/base_scan") and self.tf.frameExists("/camera_link"):
            t = self.tf_listener_.getLatestCommonTime("/base_scan", "/camera_link")
            p1 =PoseStamped()
            p1.header.frame_id = "camera_link"
            p1.pose.orientation.w = 1.0    # Neutral orientation
            p_in_base = self.tf_listener_.transformPose("/base_scan", p1)
            print(p_in_base)

        # self.tf.waitForTransform(data.header.frame_id, "/camera_link", rospy.Time.now(), rospy.Duration(2.0))
        # pp = self.tf.transformPoint("/camera_link", data.)

    # def cam_callback(self,data):
    #     crowd=data.persons
    #     print(crowd)
        
if __name__ == '__main__':
    rospy.init_node('test_node', anonymous=True)
    cls_ = Calibration()
    rospy.spin()