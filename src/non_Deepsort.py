#!/usr/bin/python3
import random
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, xywh2xyxy,strip_optimizer, set_logging,clip_boxes)
from utils.datasets import LoadStreams, LoadImages, letterbox
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch
from utils.downloads import attempt_download
from deep_sort_pytorch.utils.parser import get_config
from human_following.msg import  camera_person, camera_persons
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
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
from std_msgs.msg import Int64MultiArrayz

# PyTorch
# YoloV5-PyTorch

pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程开始
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }'''

    # 保存内参到本地
    # with open('./intrinsics.json', 'w') as fp:
    #json.dump(camera_parameters, fp)
    #######################################################

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame


class YoloV5:
    def __init__(self, yolov5_yaml_path='config/yolov5s.yaml'):
        '''初始化'''
        # 载入配置文件
        with open(yolov5_yaml_path, 'r', encoding='utf-8') as f:
            self.yolov5 = yaml.load(f.read(), Loader=yaml.SafeLoader)
        # 随机生成每个类别的颜色
        self.colors = [[np.random.randint(0, 255) for _ in range(
            3)] for class_id in range(self.yolov5['class_num'])]
        # 模型初始化
        self.init_model()

    @torch.no_grad()
    def init_model(self):
        '''模型初始化'''
        # 设置日志输出
        set_logging()
        # 选择计算设备
        device = select_device(self.yolov5['device'])
        # 如果是GPU则使用半精度浮点数 F16
        is_half = device.type != 'cpu'
        # 载入模型
        model = attempt_load(
            self.yolov5['weight'], map_location=device)  # 载入全精度浮点数的模型
        input_size = check_img_size(
            self.yolov5['input_size'], s=model.stride.max())  # 检查模型的尺寸
        if is_half:
            model.half()  # 将模型转换为半精度
        # 设置BenchMark，加速固定图像的尺寸的推理
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # 图像缓冲区初始化
        img_torch = torch.zeros(
            (1, 3, self.yolov5['input_size'], self.yolov5['input_size']), device=device)  # init img
        # 创建模型
        # run once
        _ = model(img_torch.half()
                  if is_half else img) if device.type != 'cpu' else None
        self.is_half = is_half  # 是否开启半精度
        self.device = device  # 计算设备
        self.model = model  # Yolov5模型
        self.img_torch = img_torch  # 图像缓冲区

    def preprocessing(self, img):
        '''图像预处理'''
        # 图像缩放
        # 注: auto一定要设置为False -> 图像的宽高不同
        img_resize = letterbox(img, new_shape=(
            self.yolov5['input_size'], self.yolov5['input_size']), auto=False)[0]
        # print("img resize shape: {}".format(img_resize.shape))
        # 增加一个维度
        img_arr = np.stack([img_resize], 0)
        # 图像转换 (Convert) BGR格式转换为RGB
        # 转换为 bs x 3 x 416 x
        # 0(图像i), 1(row行), 2(列), 3(RGB三通道)
        # ---> 0, 3, 1, 2
        # BGR to RGB, to bsx3x416x416
        img_arr = img_arr[:, :, :, ::-1].transpose(0, 3, 1, 2)
        # 数值归一化
        # img_arr =  img_arr.astype(np.float32) / 255.0
        # 将数组在内存的存放地址变成连续的(一维)， 行优先
        # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        # https://zhuanlan.zhihu.com/p/59767914
        img_arr = np.ascontiguousarray(img_arr)
        return img_arr

    @torch.no_grad()
    def detect(self, img, canvas=None, view_img=True):
        '''模型预测'''
        # 图像预处理
        img_resize = self.preprocessing(img)  # 图像缩放
        self.img_torch = torch.from_numpy(img_resize).to(self.device)  # 图像格式转换
        self.img_torch = self.img_torch.half(
        ) if self.is_half else self.img_torch.float()  # 格式转换 uint8-> 浮点数
        self.img_torch /= 255.0  # 图像归一化
        if self.img_torch.ndimension() == 3:
            self.img_torch = self.img_torch.unsqueeze(0)
        # 模型推理
        t1 = time_sync()
        pred = self.model(self.img_torch, augment=False)[0]
        # pred = self.model_trt(self.img_torch, augment=False)[0]
        # NMS 非极大值抑制
        pred = non_max_suppression(pred, self.yolov5['threshold']['confidence'],
                                   self.yolov5['threshold']['iou'], classes=None, agnostic=False)
        t2 = time_sync()
        # print("推理时间: inference period = {}".format(t2 - t1))
        # 获取检测结果
        det = pred[0]
        gain_whwh = torch.tensor(img.shape)[[1, 0, 1, 0]]  # [w, h, w, h]

        if view_img and canvas is None:
            canvas = np.copy(img)
            for_crop=np.copy(img)
        xyxy_list,conf_list ,class_id_list,crop_imgs = [],[],[],[]
        
        if det is not None and len(det):
            # 画面中存在目标对象
            # 将坐标信息恢复到原始图像的尺寸
            det[:, :4] = scale_coords(
                img_resize.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, class_id in reversed(det):
                if int(class_id)==0:
                    class_id = int(class_id)
                    xyxy_list.append(xyxy)
                    conf_list.append(conf)
                    class_id_list.append(class_id)
                    if view_img:
                        label = '%s %.2f' % (
                            self.yolov5['class_name'][class_id], conf)
                        self.plot_one_box(
                            xyxy, canvas, label=label, color=self.colors[class_id], line_thickness=3)
                        
                        
                        
                        xyxy=torch.tensor(xyxy).view(-1, 4)
                        b = xyxy2xywh(xyxy)        
                        # b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
                        b[:, 2:] = b[:, 2:] * 1.02 + 10  # box wh * gain + pad
                        xyxy = xywh2xyxy(b).long()
                        clip_boxes(xyxy, for_crop.shape)
                        crop = for_crop[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1)]
                        # crop=cv2.cvtColor(crop , cv2.COLOR_BGR2GRAY)
                        # print(crop)
                        crop_imgs.append(crop)
        # print(crop_imgs)

        return canvas, class_id_list, xyxy_list, conf_list,crop_imgs
    
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        ''''绘制矩形框+标签'''
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

def pub( list_1, list_2,data):
    global pub_camera_tracked
    global camera_frame
    ###### plz modified this

    crowd = camera_persons()
    bridge=CvBridge()

    while not rospy.is_shutdown():
        person = camera_person()
        if len(list_2) == 0:
            break
        else:
            person.pose.x = -list_1.pop(0) ##y
            person.pose.y = list_1.pop(0) ##depth
            person.id = list_2.pop(0)
            person.is_reconize = True
            # print(np.array(data,dtype=int).shape)
            # print(np.array(data.pop(0),dtype=int).shape)
            _crop=np.array(data.pop(0)).astype(np.uint8)  
            height,weight,dim=_crop.shape
            _crop=_crop.flatten()
            person.shape.append(height)
            person.shape.append(weight)
            person.shape.append(dim)
            person.crops=_crop
            crowd.persons.append(person)




    crowd.header.stamp = rospy.Time.now()
    crowd.header.frame_id = camera_frame
    pub_camera_tracked.publish(crowd)

def image_pub(data):
    global pub_camera_img
    bridge=CvBridge()
    
    img_msg=bridge.cv2_to_imgmsg(data)
    pub_camera_img.publish(img_msg)
    



    



if __name__ == '__main__':
    print("[INFO] YoloV5目标检测-程序启动")
    print("[INFO] 开始YoloV5模型加载")
    # YOLOV5模型配置文件(YAML格式)的路径 yolov5_yaml_path
    model = YoloV5(yolov5_yaml_path='/home/cai/catkin_ws/src/human_following/src/config/yolov5s.yaml')
    print("[INFO] 完成YoloV5模型加载")
    rospy.init_node('yolov5_node')
    global pub_camera_tracked
    global camera_frame
    global pub_camera_img
    global pub_crops


    camera_frame = rospy.get_param("camera_frame", "camera_link")
    pub_camera_tracked = rospy.Publisher("tracker/data", camera_persons, queue_size=1)
    pub_camera_img = rospy.Publisher("tracker/image", Image, queue_size=1)

    try:
        while not rospy.is_shutdown():
            # Wait for a coherent pair of frames: depth and color
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
            if not depth_image.any() or not color_image.any():
                continue
            # Convert images to numpy arrays
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            

            t_start = time.time()  
            

            canvas, class_id_list, xyxy_list, conf_lis,crop_imgs= model.detect(color_image)

            t_end = time.time()  # 结束计时\
            new_class_id_list,pub_axis_list=list(),list()
            if xyxy_list:
                for i  in range(len(class_id_list)):
                        ux = int((xyxy_list[i][0]+xyxy_list[i][2])/2)  # 计算像素坐标系的x
                        uy = int((xyxy_list[i][1]+xyxy_list[i][3])/2)  # 计算像素坐标系的y
                        dis = aligned_depth_frame.get_distance(ux, uy)
                        camera_xyz = rs.rs2_deproject_pixel_to_point(
                            depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz
                        camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                        camera_xyz = camera_xyz.tolist()
                        new_class_id_list.append(class_id_list[i])
                        pub_axis_list.append(camera_xyz[0])
                        pub_axis_list.append(camera_xyz[2])
                        cv2.circle(canvas, (ux,uy), 4, (255, 255, 255), 5)#标出中心点
                        cv2.putText(canvas, str(camera_xyz), (ux+20, uy+10), 0, 1,
                                    [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)#标出坐标

            ### pub_axis_list, pub_id_list,class_id_list
            # [[x,z], [id], [class(0/1)]]


            pub( pub_axis_list, new_class_id_list,crop_imgs)
            image_pub(canvas)

    finally:
        # Stop streaming
        cv2.destroyAllWindows()
        pipeline.stop()
