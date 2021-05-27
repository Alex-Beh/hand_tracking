#!/usr/bin/env python3.6

#Quick fix
# import sys
# sys.path.insert(0, '..')

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import numpy as np

from hand_tracking.models.experimental import attempt_load
from hand_tracking.utils.datasets import LoadStreams, LoadImages,letterbox
from hand_tracking.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from hand_tracking.utils.plots import plot_one_box
from hand_tracking.utils.torch_utils import select_device, load_classifier, time_synchronized

import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

class GestureRecogniton:
    def __init__(self,opt,save_img=False):
        source, weights, self.view_img, self.save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))
        
        self.save_img = save_img

        # Directories
        # self.save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        # (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(opt.device)

        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        #"Fusing layers..."

        # Second-stage classifier
        self.classify = False
        if self.classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once   

        # ROS
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/gesture_detection_image', Image,queue_size=10)
        # rgb_image_sub = rospy.Subscriber("/spencer/sensors/rgbd_front_top/color/image_raw", Image, self.image_callback)

        rgb_image_sub = message_filters.Subscriber("/spencer/sensors/rgbd_front_top/color/image_raw", Image)
        # depth_image_sub = message_filters.Subscriber('/spencer/sensors/rgbd_front_top/depth/image_rect_raw', Image)
        
        depth_image_sub = message_filters.Subscriber('/spencer/sensors/rgbd_front_top/aligned_depth_to_color/image_raw', Image)

        cmd_vel_sub = message_filters.Subscriber('/tb_control/wheel_odom', Odometry)
        self.time_sync = message_filters.ApproximateTimeSynchronizer([rgb_image_sub, depth_image_sub,cmd_vel_sub], 10, 1)
        # self.time_sync = message_filters.ApproximateTimeSynchronizer([rgb_image_sub, depth_image_sub], 5, 0.2)

        self.time_sync.registerCallback(self.image_callback)

        self.gesture_pub = rospy.Publisher('/gesture_recognition_result', Header, queue_size=10)

        self.distance_threshold = 450
        self.conf_threshold = 0.7
        self.last_gesture = None
        self.last_detect_time = 0
        self.timeout = rospy.Duration(3)


    def image_callback(self,rgb_img_msg,depth_img_msg,odom_msg):
        print("!!!")
        if(abs(odom_msg.twist.twist.linear.x)>0.01 and abs(odom_msg.twist.twist.angular.z)>0.01):
            print("Only run hand detection model when the robot is not moving")
            return
    # def image_callback(self,rgb_img_msg,depth_img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_img_msg, "passthrough")
            depth_image = self.bridge.imgmsg_to_cv2(depth_img_msg,"passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)               # shape: (480, 640)

        except CvBridgeError as e:
            print(e)
            return 


        img = torch.from_numpy(cv_image).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # swap cahnnel
        img = img.permute(0,3,1,2)

        print(img.shape)
        pred = self.model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        
        print("--")
        # Process detections                   
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(cv_image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img .shape[2:], det[:, :4], cv_image.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_img or self.view_img:  # Add bbox to image
                        c1, c2 = int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)
                        depth_handmidpoint = depth_array[c2,c1]
                        print(depth_handmidpoint)
                        label = '%s %.2f' % (self.names[int(cls)], conf)


                        if conf >= self.conf_threshold and depth_handmidpoint<self.distance_threshold:
                            gesture_result_msg = Header()
                            gesture_result_msg.stamp = rospy.Time.now()
                            if self.last_gesture ==None:
                                self.last_gesture = label.split(" ")[0]
                                self.last_detect_time = rospy.Time.now()
                                gesture_result_msg.frame_id = label.split(" ")[0]
                            elif self.last_gesture!=label.split(" ")[0] or rospy.Time.now()-self.last_detect_time>self.timeout:
                                self.last_gesture = label.split(" ")[0]
                                self.last_detect_time = rospy.Time.now()
                                gesture_result_msg.frame_id = label.split(" ")[0]
                            print("Result: {}".format(gesture_result_msg.frame_id))
                            self.gesture_pub.publish(gesture_result_msg)
                        # plot_one_box(xyxy, cv_image, label=label, color=self.colors[int(cls)], line_thickness=3)
                            
            # Stream results
            # self.pub.publish(self.bridge.cv2_to_imgmsg(cv_image,encoding="rgb8"))

def detect_one(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    #---
    img0 = cv2.imread("1234.jpg")  # BGR
    
    print(img0.shape)

    img = letterbox(img0, 640, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    print(type(img))
    print(img.shape)

    pred = model(img, augment=opt.augment)[0]
    # print(pred)

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img .shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if conf >= 0.6:
                    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
                    plot_one_box(xyxy, img0, label="label", line_thickness=3)
                    cv2.imwrite("check1234.jpg",img0)

if __name__ == '__main__':
    rospy.init_node('hand_gesture_recognition', anonymous=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='../runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    
    opt.source = "0"
    # opt.img_size = 320
    opt.weights = "/home/followme/leg_detector/src/hand_tracking/runs/train/exp16_gesture_dataset/weights/best.pt"
    opt.view_img = True
    opt.agnostic_nms = True

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            gesture_recognizer = GestureRecogniton(opt)
            # detect_one()

            try:
                rospy.spin()
            except KeyboardInterrupt:
                print("Shutting down")
