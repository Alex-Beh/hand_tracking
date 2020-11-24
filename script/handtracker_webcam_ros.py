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

from hand_tracker import HandTracker
import gesture_recognition

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import message_filters

from cv_bridge import CvBridge, CvBridgeError

class handtracker_webcam:
    def __init__(self):
        print("Initialize handtracker_webcam")
        self.collection_length = 10
        self.match_threshold   = 3
        self.gesture_recognition_history = dict()

        self.checked_gesture = False
        self.last_gesture    = None

        palm_model_path = "/home/alex-beh/ros1_ws/catkin_ws/src/hand_tracking/models/palm_detection.tflite"
        landmark_model_path = "/home/alex-beh/ros1_ws/catkin_ws/src/hand_tracking/models/hand_landmark.tflite"
        anchors_path = "/home/alex-beh/ros1_ws/catkin_ws/src/hand_tracking/data/anchors.csv"

        self.detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                                box_shift=0.2, box_enlarge=1.3)

        self.bridge = CvBridge()
        self.gesture_pub = rospy.Publisher('/gesture_recognition_result', Header, queue_size=10)
        self.result_img_pub = rospy.Publisher('/gesture_recognition_result_image', Image, queue_size=10)

        '''
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.image_callback)
        '''
        rgb_image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        depth_image_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        # rgb_image_sub = message_filters.Subscriber("/spencer/sensors/rgbd_front_top/color/image_raw", Image)
        # depth_image_sub = message_filters.Subscriber('/spencer/sensors/rgbd_front_top/depth/image_rect_raw', Image)
        self.time_sync = message_filters.ApproximateTimeSynchronizer([rgb_image_sub, depth_image_sub], 1, 0.05)
        self.time_sync.registerCallback(self.image_callback)

        self.publish_result = True

    def image_callback(self,rgb_img_msg,depth_img_msg):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_img_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_img_msg,"passthrough")
           
            print(cv_image.shape)
            print(depth_image.shape)

            depth_array = np.array(depth_image, dtype=np.float32)

        except CvBridgeError as e:
            print(e)
            return 

        image = cv_image[:,:,::-1]
        kp, box = self.detector(image)

        if kp is not None:
            area = sqrt(((box[0][0]-box[3][0])**2+(box[0][1]-box[3][1])**2)) * sqrt(((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2))
            print(kp)
            print(kp.flatten())
            print(int(kp.flatten()[0]),int(kp.flatten()[1]))
            depth_handmidpoint = depth_array[int(kp.flatten()[0]),int(kp.flatten()[1])]
            print(depth_handmidpoint)
            if(area<50000):
                return
                
            try:
                # depth_handmidpoint = depth_array[int(kp.flatten()[0]),int(kp.flatten()[1])]
                
                # if(depth_handmidpoint < 650):

                self.draw_hand(cv_image, kp)
                self.draw_box(cv_image, box)
            except:
                return
            # else:
                # rospy.loginfo("Depth to hand midpoint: {}".format(depth_handmidpoint))

            if self.publish_result:
                try:
                    cv2.putText(cv_image, "0",(int(box[0][0]),int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
                    cv2.putText(cv_image, "1",(int(box[1][0]),int(box[1][1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
                    cv2.putText(cv_image, "2",(int(box[2][0]),int(box[2][1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
                    cv2.putText(cv_image, "3",(int(box[3][0]),int(box[3][1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
                    cv2.circle(cv_image, (int(kp.flatten()[0]), int(kp.flatten()[1])), 3, (0,0,255), -1)
                    self.result_img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
                except CvBridgeError as e:
                    print(e)
                    return

    def draw_hand(self,frame, kp):

        gesture_result = gesture_recognition.gesture_recognition(kp.flatten())

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
            gesture_result = "Pending"

        print(gesture_result)
        gesture_result_msg = Header()
        gesture_result_msg.stamp = rospy.Time.now()
        gesture_result_msg.frame_id = gesture_result
        self.gesture_pub.publish(gesture_result_msg)

        if self.publish_result:
            # the points where we go back to the first midpoint of the hand
            clearpoints = [4,8,12,16,20]
            lk = None
            p = 0

            # Draw the hand
            cv2.putText(frame, gesture_result , (100,100), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 

            for keypoint in kp:
                if lk is not None:
                    cv2.line(frame, (int(keypoint[0]),int(keypoint[1])),(int(lk[0]),int(lk[1])), (255,0,255), 2)
                lk = keypoint
                cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, (0,255,255), -1)

                if p in clearpoints:
                    lk = kp[0]
                p = p + 1

    def draw_box(self,frame, box):
        # draw the box
        for i in range(0,4):
            cv2.line(frame, (int(box[i][0]),int(box[i][1])),(int(box[(i+1)&3][0]),int(box[(i+1)&3][1])), (255,255,255), 2)

if __name__ == '__main__':
    rospy.init_node('hand_gesture_recognition', anonymous=True)

    ic = handtracker_webcam()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

