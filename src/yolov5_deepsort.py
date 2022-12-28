#!/usr/bin/python3
import random
from utils.torch_utils import select_device, time_sync
from utils.general import (
    check_img_size, non_max_suppression, scale_coords,
    xyxy2xywh, xywh2xyxy, set_logging,clip_boxes)
from utils.datasets import letterbox
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch
from utils.downloads import attempt_download
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from human_following.msg import  camera_person, camera_persons
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import yaml
import time
import numpy as np
import rospy
import cv2
import os
# PyTorch
# YoloV5-PyTorch

pipeline = rs.pipeline()  # å®šä¹‰æµç¨‹pipeline
config = rs.config()  # å®šä¹‰é…ç½®config
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
profile = pipeline.start(config)  # æµç¨‹å¼€å§‹
align_to = rs.stream.color  # ä¸Žcoloræµå¯¹é½
align = rs.align(align_to)



def get_aligned_images():
    frames = pipeline.wait_for_frames()  # ç­‰å¾…èŽ·å–å›¾åƒå¸§
    aligned_frames = align.process(frames)  # èŽ·å–å¯¹é½å¸§
    aligned_depth_frame = aligned_frames.get_depth_frame()  # èŽ·å–å¯¹é½å¸§ä¸­çš„depthå¸§
    color_frame = aligned_frames.get_color_frame()  # èŽ·å–å¯¹é½å¸§ä¸­çš„colorå¸§

    ############### ç›¸æœºå‚æ•°çš„èŽ·å– #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # èŽ·å–ç›¸æœºå†…å‚
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics  # èŽ·å–æ·±åº¦å‚æ•°ï¼ˆåƒç´ åæ ‡ç³»è½¬ç›¸æœºåæ ‡ç³»ä¼šç”¨åˆ°ï¼‰
    '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }'''

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # æ·±åº¦å›¾ï¼ˆé»˜è®¤16ä½ï¼‰
    # depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # æ·±åº¦å›¾ï¼ˆ8ä½ï¼‰
    # depth_image_3d = np.dstack(
    #     (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3é€šé“æ·±åº¦å›¾
    color_image = np.asanyarray(color_frame.get_data())  # RGBå›¾

    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame


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
        self.cfg = get_config()
        self.cfg.merge_from_file("/home/cai/catkin_ws/src/human_following/src/deep_sort_pytorch/configs/deep_sort.yaml")
        attempt_download("/home/cai/catkin_ws/src/human_following/src/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        self.deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                    max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                    max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT,
                    nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

    @torch.no_grad()
    def init_model(self):
        '''æ¨¡åž‹åˆå§‹åŒ–'''
        # è®¾ç½®æ—¥å¿—è¾“å‡º
        set_logging()
        # é€‰æ‹©è®¡ç®—è®¾å¤‡
        device = select_device(self.yolov5['device'])
        # å¦‚æžœæ˜¯GPUåˆ™ä½¿ç”¨åŠç²¾åº¦æµ®ç‚¹æ•° F16
        is_half = device.type != 'cpu'
        # è½½å…¥æ¨¡åž‹
        model = attempt_load(
            self.yolov5['weight'], map_location=device)  # è½½å…¥å…¨ç²¾åº¦æµ®ç‚¹æ•°çš„æ¨¡åž‹
        self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names

        # input_size = check_img_size(
        #     self.yolov5['input_size'], s=model.stride.max())  # æ£€æŸ¥æ¨¡åž‹çš„å°ºå¯¸
        if is_half:
            model.half()  # å°†æ¨¡åž‹è½¬æ¢ä¸ºåŠç²¾åº¦
        # è®¾ç½®BenchMarkï¼ŒåŠ é€Ÿå›ºå®šå›¾åƒçš„å°ºå¯¸çš„æŽ¨ç†
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # å›¾åƒç¼“å†²åŒºåˆå§‹åŒ–
        img_torch = torch.zeros(
            (1, 3, self.yolov5['input_size'], self.yolov5['input_size']), device=device)  # init img
        # åˆ›å»ºæ¨¡åž‹
        # run once
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
            
        xyxy_list,conf_list ,class_id_list,crop_imgs = [],[],[],[]
        
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_resize.shape[2:], det[:, :4], img.shape).round()

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), canvas)
            for _, (output, conf) in enumerate(zip(outputs, confs)):
                if int(output[5])==0:
                    class_id = int(output[5])
                    xyxy_list.append(output[0:4])
                    conf_list.append(conf)
                    class_id_list.append(output[4])
                    label = f'{int(output[5])} {self.names[class_id]} {conf:.2f}'
                    self.plot_one_box(output[0:4], canvas, label=label, color=self.colors[class_id], line_thickness=3)
                    xyxy=torch.tensor(output[0:4]).view(-1, 4)
                    b = xyxy2xywh(xyxy)        
                    b[:, 2:] = b[:, 2:] * 1.02 + 10  # box wh * gain + pad
                    xyxy = xywh2xyxy(b).long()
                    clip_boxes(xyxy, canvas.shape)
                    crop = canvas[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1)]
                    crop_imgs.append(crop)   
        return canvas, class_id_list, xyxy_list, conf_list,crop_imgs

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        ''''ç»˜åˆ¶çŸ©å½¢æ¡†+æ ‡ç­¾'''

        if label:
            tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            color = color or [random.randint(0, 255) for _ in range(3)]
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
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

    crowd = camera_persons()

    while not rospy.is_shutdown():
        person = camera_person()
        if len(list_2) == 0:
            break
        else:
            person.pose.x = -list_1.pop(0) ##y
            person.pose.y = list_1.pop(0) ##depth
            person.id = list_2.pop(0)
            person.is_reconize = True

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
    
    model = YoloV5(yolov5_yaml_path=os.path.dirname(__file__) +'/config/yolov5s.yaml')
    rospy.init_node('yolov5_deepsort_node')

    global pub_camera_tracked
    global camera_frame
    global pub_camera_img
    global pub_crops
    camera_frame = rospy.get_param("camera_frame", "camera_link")
    pub_camera_tracked = rospy.Publisher("tracker/data", camera_persons, queue_size=1)
    pub_camera_img = rospy.Publisher("tracker/image", Image, queue_size=1)

    try:
        while not rospy.is_shutdown():
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()  # èŽ·å–å¯¹é½çš„å›¾åƒä¸Žç›¸æœºå†…å‚
            if not depth_image.any() or not color_image.any():
                rospy.sleep(0.01)
                continue

            canvas, class_id_list, xyxy_list, conf_lis,crop_imgs= model.detect(color_image)
           
            new_class_id_list,pub_axis_list=list(),list()
            try:
                if xyxy_list:
                    for i ,index in enumerate(class_id_list):
                        ux = int((xyxy_list[i][0]+xyxy_list[i][2])/2)  # è®¡ç®—åƒç´ åæ ‡ç³»çš„x
                        uy = int((xyxy_list[i][1]+xyxy_list[i][3])/2)  # è®¡ç®—åƒç´ åæ ‡ç³»çš„y
                        dis = aligned_depth_frame.get_distance(ux, uy)
                        camera_xyz = rs.rs2_deproject_pixel_to_point(
                            depth_intrin, (ux, uy), dis)  # è®¡ç®—ç›¸æœºåæ ‡ç³»çš„xyz
                        camera_xyz = np.round(np.array(camera_xyz), 3)  # è½¬æˆ3ä½å°æ•°
                        camera_xyz = camera_xyz.tolist()
                        new_class_id_list.append(class_id_list[i])
                        pub_axis_list.append(camera_xyz[0])
                        pub_axis_list.append(camera_xyz[2])
                        # cv2.circle(canvas, (ux,uy), 4, (255, 255, 255), 5)#æ ‡å‡ºä¸­å¿ƒç‚¹
                        # cv2.putText(canvas, str(camera_xyz), (ux+20, uy+10), 0, 1,
                        #             [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)#æ ‡å‡ºåæ ‡
            except:
                pass
            ### pub_axis_list, pub_id_list,class_id_list
            # [[x,z], [id], [class(0/1)]]
            pub( pub_axis_list, new_class_id_list,crop_imgs)
            image_pub(canvas)
            rospy.sleep(0.005)

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
