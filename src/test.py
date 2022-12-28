#!/usr/bin/python2
import random
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.datasets import LoadStreams, LoadImages, letterbox
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch
from utils.downloads import attempt_download
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from human_following.msg import  camera_person, camera_persons
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import pyrealsense2 as rs
import math
import yaml
import argparse
import os
import time
import numpy as np
import sys
import rospy
import cv2
# PyTorch
# YoloV5-PyTorch

class YoloV5:
    def __init__(self, yolov5_yaml_path='config/yolov5s.yaml'):
        '''åˆå§‹åŒ–'''
        # è½½å…¥é…ç½®æ–‡ä»¶
        with open(yolov5_yaml_path, 'r', encoding='utf-8') as f:
            self.yolov5 = yaml.load(f.read(), Loader=yaml.SafeLoader)
        # éšæœºç”Ÿæˆæ¯ä¸ªç±»åˆ«çš„é¢œè‰²
        self.colors = [[np.random.randint(0, 255) for _ in range(
            3)] for class_id in range(self.yolov5['class_num'])]
        # æ¨¡åž‹åˆå§‹åŒ–
        self.init_model()

    # ____ deepsort ____ #
        self.cfg = get_config()
        self.cfg.merge_from_file("/home/cai/catkin_ws/src/human_following/src/deep_sort_pytorch/configs/deep_sort.yaml")
        attempt_download("/home/cai/catkin_ws/src/human_following/src/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        self.deepsort=DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                    max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                    max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT,
                    nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
    # ____ deepsort ____ #

    # ____ realsense ____ #
        # self.intr, self.depth_intrin, self.color_image, self.depth_image, self.aligned_depth_frame=None,None,None,None,None
        
        # self.pipeline = rs.pipeline()  # å®šä¹‰æµç¨‹pipeline
        # self.config = rs.config()  # å®šä¹‰é…ç½®config
        # self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        # self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        # self.profile = self.pipeline.start(self.config)  # æµç¨‹å¼€å§‹
        # self.align_to = rs.stream.color  # ä¸Žcoloræµå¯¹é½
        # self.align = rs.align(self.align_to)
    # ____ realsense ____ #


    # ____ publish ____ #
        self.pubs = {}
        self.pubs['yolov5_deepsort/data'] = rospy.Publisher("yolov5_deepsort/data", Image,queue_size=1)
        self.pubs['yolov5_deepsort/image'] = rospy.Publisher("yolov5_deepsort/image", Image,queue_size=1)
        self.camera_frame=rospy.get_param("camera_frame", "camera_link")
        self.bridge=CvBridge()
    # ____ publish ____ #

    # ____ subscribe ____ #
        self.sub1=[rospy.Subscriber('/camera/color/image_raw', Image, self.rs_color_image)]
        self.sub2=[rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.rs_depth_image)]
        self.sub3=[rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.rs_color_info)]
        self.sub4=[rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.rs_depth_info)]
    # ____ subscribe ____ #

    

    @torch.no_grad()
    def init_model(self):
        '''æ¨¡åž‹åˆå§‹åŒ–'''
        set_logging()
        device = select_device(self.yolov5['device'])
        is_half = device.type != 'cpu'
        model = attempt_load(
            self.yolov5['weight'], map_location=device)  
        self.names = model.module.names if hasattr(model, 'module') else model.names  

        input_size = check_img_size(
            self.yolov5['input_size'], s=model.stride.max())  
        if is_half:
            model.half()  
        cudnn.benchmark = True  

        img_torch = torch.zeros(
            (1, 3, self.yolov5['input_size'], self.yolov5['input_size']), device=device)  # init img
       
        _ = model(img_torch.half()
                  if is_half else img) if device.type != 'cpu' else None
        self.is_half = is_half  # æ˜¯å¦å¼€å¯åŠç²¾åº¦
        self.device = device  # è®¡ç®—è®¾å¤‡
        self.model = model  # Yolov5æ¨¡åž‹
        self.img_torch = img_torch  # å›¾åƒç¼“å†²åŒº

    def preprocessing(self, img):
        '''å›¾åƒé¢„å¤„ç†'''
        img_resize = letterbox(img, new_shape=(
            self.yolov5['input_size'], self.yolov5['input_size']), auto=False)[0]
      
        img_arr = np.stack([img_resize], 0)
        img_arr = img_arr[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img_arr = np.ascontiguousarray(img_arr)
        return img_arr

    @torch.no_grad()
    def detect(self, img, canvas=None, view_img=True):
        '''æ¨¡åž‹é¢„æµ‹'''
        img_resize = self.preprocessing(img)  # å›¾åƒç¼©æ”¾
        self.img_torch = torch.from_numpy(img_resize).to(self.device)  # å›¾åƒæ ¼å¼è½¬æ¢
        self.img_torch = self.img_torch.half(
        ) if self.is_half else self.img_torch.float()  # æ ¼å¼è½¬æ¢ uint8-> æµ®ç‚¹æ•°
        self.img_torch /= 255.0  # å›¾åƒå½’ä¸€åŒ–
        if self.img_torch.ndimension() == 3:
            self.img_torch = self.img_torch.unsqueeze(0)
        t1 = time_sync()
        pred = self.model(self.img_torch, augment=False)[0]
        pred = non_max_suppression(pred, self.yolov5['threshold']['confidence'],
                                   self.yolov5['threshold']['iou'], classes=None, agnostic=False)
        t2 = time_sync()
        det = pred[0]

        if view_img and canvas is None:  canvas = np.copy(img)

        xyxy_list,conf_list,class_id_list,pub_id_list= list(),list(),list(),list()
        
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_resize.shape[2:], det[:, :4], img.shape).round()
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), canvas)
            if len(outputs)>0:
        
                for _, (output, conf) in enumerate(zip(outputs, confs)):
                    class_id = int(output[5])
                    pub_id_list.append(class_id)
                    xyxy_list.append(output[0:4])
                    conf_list.append(conf)
                    class_id_list.append(output[4])
                    for id in class_id_list:
                        if view_img:
                            # ç»˜åˆ¶çŸ©å½¢æ¡†ä¸Žæ ‡ç­¾
                            if class_id==0:
                                label = f'{id} {self.names[class_id]} {conf:.2f}'
                                self.plot_one_box(output[0:4], canvas, label=label, color=self.colors[class_id], line_thickness=3)
                                
        return canvas, class_id_list, xyxy_list, conf_list,pub_id_list

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        ''''ç»˜åˆ¶çŸ©å½¢æ¡†+æ ‡ç­¾'''
        
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def rs_depth_image(self,data):
        self.aligned_depth_frame=self.bridge.imgmsg_to_cv2(data, deside_encoding="passthrough")
        self.depth_image=np.asanyarray(self.aligned_depth_frame)
        depth_image_8bit = cv2.convertScaleAbs(self.depth_image, alpha=0.03)  
        depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))  

    def rs_color_image(self,data):
        self.color_frame=self.bridge.imgmsg_to_cv2(data, deside_encoding="passthrough")
        self.color_image = np.asanyarray(self.color_frame)  
    
    def rs_depth_info(self,data):
        self.depth_intrin = data  

    def rs_color_info(self,data):
        self.intr = data  
        '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }'''

    def yolo_deepsort(self):
        if not self.depth_image.any() or not self.color_image.any():
            pass    
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((self.color_image, depth_colormap))
        canvas, class_id_list, xyxy_list, _,pub_id_list=self.detect(self.color_image)
        new_class_id_list,camera_xyz_list,pub_axis_list=list(),list(),list()
        if xyxy_list:
                for i ,index in enumerate(pub_id_list):
                    if index==0:
                        ux = int((xyxy_list[i][0]+xyxy_list[i][2])/2) 
                        uy = int((xyxy_list[i][1]+xyxy_list[i][3])/2) 
                        dis = self.aligned_depth_frame.get_distance(ux, uy)
                        camera_xyz = rs.rs2_deproject_pixel_to_point(
                            self.depth_intrin, (ux, uy), dis)  # è®¡ç®—ç›¸æœºåæ ‡ç³»çš„xyz
                        camera_xyz = np.round(np.array(camera_xyz), 3)  # è½¬æˆ3ä½å°æ•°
                        camera_xyz = camera_xyz.tolist()
                        new_class_id_list.append(class_id_list[i])
                        pub_axis_list.append(camera_xyz[0])
                        pub_axis_list.append(camera_xyz[1])
                        cv2.circle(canvas, (ux,uy), 4, (255, 255, 255), 5)
                        cv2.putText(canvas, str(camera_xyz), (ux+20, uy+10), 0, 1,
                                    [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)#æ ‡å‡ºåæ ‡
                        camera_xyz_list.append(camera_xyz)
        
        self.data_pub(pub_axis_list, new_class_id_list)
        self.image_pub(canvas)

    def data_pub(self,list_1, list_2):
        persons = camera_persons()
        while not rospy.is_shutdown():
            person = camera_person()
            if len(list_2) == 0:
                break
            else:
                person.pose.x = -list_1.pop(0) ##y
                person.pose.y = list_1.pop(0) ##depth
                person.id = list_2.pop(0)
                person.is_reconize = True
                persons.persons.append(person)
                
        persons.header.stamp = rospy.Time.now()
        persons.header.frame_id = self.camera_frame
        self.pubs['yolov5_deepsort/data'].publish(persons)        

    def image_pub(self,image):
        img_msg=self.bridge.cv2_to_imgmsg(image)     
        self.pubs['yolov5_deepsort/image'].publish(img_msg)


if __name__ == '__main__':
    rospy.init_node('yolov5_deepsort_node')
    model = YoloV5(yolov5_yaml_path='/home/cai/catkin_ws/src/human_following/src/config/yolov5s.yaml')
    rospy.spin()
