#!/usr/bin/env python3
#
# Basic example of how to use the handtracker using a web camera as input.
#
# Author: Joakim Eriksson, joakim.eriksson@ri.se
#
import cv2
import numpy as np
import collections
from math import sqrt

from detector.hand_tracker import HandTracker
import gesture_recognition
from gestureDetector import gestureDetector

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import message_filters

from cv_bridge import CvBridge, CvBridgeError

kp_ = None
box_ = None

class gestureRecognitionTFLite(gestureDetector):
    def __init__(self,name,detector,confidence_threshold,image_height,image_width):
        gestureDetector.__init__(self,name,detector,confidence_threshold,image_height,image_width)

    def detect_image(self,imageInput):
        global kp_, box_
        kp, box = self.detector(imageInput)
        if kp is not None:
            kp_ = kp
            box_ = box
            area = sqrt(((box[0][0]-box[3][0])**2+(box[0][1]-box[3][1])**2)) * sqrt(((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2))

            # if(area<50000):
            #     return False, "Not result" , 0 , 0 ,0
            try:
                gesture_result = gesture_recognition.gesture_recognition(kp.flatten())
                return True,gesture_result,100, 0 , 0
            except:
                print("gg")
                return False, "Not result" , kp , box ,0        
        else:
            return False, "Not result" , 0 , 0 ,0        


class handtracker_webcam:
    def __init__(self):
        print("Initialize handtracker_webcam")
        self.collection_length = 10
        self.match_threshold   = 3
        self.gesture_recognition_history = dict()

        self.checked_gesture = False
        self.last_gesture    = None

        palm_model_path = "/home/followme/followme_ws/src/hand_tracking/script/detector/tflite_models/palm_detection.tflite"
        landmark_model_path = "/home/followme/followme_ws/src/hand_tracking/script/detector/tflite_models/hand_landmark.tflite"    
        anchors_path = "/home/followme/followme_ws/src/hand_tracking/example/anchors.csv"

        self.detector = gestureRecognitionTFLite("gestureRecognitionTFLite",detector=HandTracker(palm_model_path, landmark_model_path, anchors_path, \
                                box_shift=0.2, box_enlarge=1.3),confidence_threshold=0.6,image_height=480,image_width=640)

        self.bridge = CvBridge()
        self.gesture_pub = rospy.Publisher('/gesture_recognition_result', Header, queue_size=10)
        self.result_img_pub = rospy.Publisher('/gesture_recognition_result_image', Image, queue_size=10)

        '''
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.image_callback)
        '''
        # rgb_image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        # depth_image_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        rgb_image_sub = message_filters.Subscriber("/spencer/sensors/rgbd_front_top/color/image_raw", Image)
        depth_image_sub = message_filters.Subscriber('/spencer/sensors/rgbd_front_top/depth/image_rect_raw', Image)
        self.time_sync = message_filters.ApproximateTimeSynchronizer([rgb_image_sub, depth_image_sub], 1, 0.05)
        self.time_sync.registerCallback(self.image_callback)

        self.publish_result = True

    def image_callback(self,rgb_img_msg,depth_img_msg):
        global kp, box

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_img_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_img_msg,"passthrough")
           
            depth_array = np.array(depth_image, dtype=np.float32)

        except CvBridgeError as e:
            print(e)
            return 

        flag_result, gesture_result ,_ , _ , _ = self.detector.detect_image(cv_image)

        if gesture_result in self.gesture_recognition_history:
            self.gesture_recognition_history[gesture_result].append(1)
        else:
            match_history = collections.deque(maxlen=self.collection_length)
            match_history.append(1)
            self.gesture_recognition_history[gesture_result] = match_history

            for key in self.gesture_recognition_history.keys():
                if key == gesture_result:
                    continue
                self.gesture_recognition_history[key].append(0)
        
        self.checked_gesture = False
    
        for key in self.gesture_recognition_history.keys():
            if not self.checked_gesture:
                if(sum(self.gesture_recognition_history[key]) >= self.match_threshold):
                    # print("Match over {} in {}".format(self.match_threshold,len(self.gesture_recognition_history[key])))
                    self.checked_gesture = True         
                    self.last_gesture    = key
                    self.gesture_recognition_history = dict()
                    break

        if not self.checked_gesture and gesture_result != self.last_gesture:
            gesture_result = "Pending + {}".format(gesture_result)

        print(gesture_result)
        gesture_result_msg = Header()
        gesture_result_msg.stamp = rospy.Time.now()
        gesture_result_msg.frame_id = gesture_result
        self.gesture_pub.publish(gesture_result_msg)

        if flag_result:

            if self.publish_result:
                try:

                    self.draw_box(cv_image,box_)

                    cv2.putText(cv_image, "0",(int(box_[0][0]),int(box_[0][1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
                    cv2.putText(cv_image, "1",(int(box_[1][0]),int(box_[1][1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
                    cv2.putText(cv_image, "2",(int(box_[2][0]),int(box_[2][1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
                    cv2.putText(cv_image, "3",(int(box_[3][0]),int(box_[3][1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 

                    clearpoints = [4,8,12,16,20]
                    lk = None
                    p = 0

                    # Draw the hand
                    cv2.putText(cv_image, gesture_result , (100,100), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 

                    if(kp_[0][0]>480):
                        cv2.circle(cv_image, (int(kp_.flatten()[0]), int(kp_.flatten()[1])), 3, (0,0,255), -1)
                    else:
                        print("Here {}".format(kp_[0][0]))
                    
                    for keypoint in kp_:
                        if lk is not None:
                            cv2.line(cv_image, (int(keypoint[0]),int(keypoint[1])),(int(lk[0]),int(lk[1])), (255,0,255), 2)
                        lk = keypoint
                        cv2.circle(cv_image, (int(keypoint[0]), int(keypoint[1])), 3, (0,255,255), -1)

                        if p in clearpoints:
                            lk = kp_[0]
                        p = p + 1

                    self.result_img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

                except CvBridgeError as e:
                    print(e)
                    return
            else:
                self.result_img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def draw_box(self,frame, box):
        for i in range(0,4):
            cv2.line(frame, (int(box[i][0]),int(box[i][1])),(int(box[(i+1)&3][0]),int(box[(i+1)&3][1])), (255,255,255), 2)

if __name__ == '__main__':
    rospy.init_node('hand_gesture_recognition', anonymous=True)

    ic = handtracker_webcam()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

