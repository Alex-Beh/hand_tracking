#!/usr/bin/env python3
#
# Basic example of how to use the handtracker using a web camera as input.
#
# Author: Joakim Eriksson, joakim.eriksson@ri.se
#
import cv2
import time
import numpy as np
import collections

import gesture_recognition
from gestureDetector import gestureDetector
from detector.net import HandKeypoint

import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

class keypointGestureDetector(gestureDetector):
    def __init__(self,name,detector,confidence_threshold,image_height,image_width):
        gestureDetector.__init__(self,name,detector,confidence_threshold,image_height,image_width)

    def detect_image(self,imageInput):
        many_keypoints, _ = self.detector.pyramid_inference(imageInput)

        if many_keypoints != []:
            for _, points in enumerate(many_keypoints):
                gesture_result = "Pending"

                point_ = np.array([])
                valid_input = True
                for i in points:
                    if i == []:
                        valid_input = False
                        continue
                    point_ = np.append(point_,int(i[0]))
                    point_ = np.append(point_,int(i[1]))

                if(valid_input):
                    if not (points[0][0]<self.image_height and points[0][1]<self.image_width):
                        print("Something is wrong in the hand keypoint detection result.")
                        return 

                    gesture_result = gesture_recognition.gesture_recognition(point_.flatten())
                    return True, gesture_result, 1 , points[0][0] , points[0][1]
                else:
                    return False, "Not result" , 0 , 0 ,0        
        else:
            return False, "Not result" , 0 , 0 ,0


class handtracker_webcam:
    def __init__(self):
        print("Initialize handtracker_webcam")
        self.collection_length = 8
        self.match_threshold   = 3
        self.gesture_recognition_history = dict()

        self.distance_threshold = 12000

        self.checked_gesture = False
        self.last_gesture    = None

        self.detector = keypointGestureDetector("keypointGestureDetector",detector=HandKeypoint(),confidence_threshold=0.6,image_height=480,image_width=640)

        self.bridge = CvBridge()
        self.gesture_pub = rospy.Publisher('/gesture_recognition_result', Header, queue_size=10)

        # rgb_image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        # depth_image_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        rgb_image_sub = message_filters.Subscriber("/spencer/sensors/rgbd_front_top/color/image_raw", Image)
        depth_image_sub = message_filters.Subscriber('/spencer/sensors/rgbd_front_top/depth/image_rect_raw', Image)
        cmd_vel_sub = message_filters.Subscriber('/odom', Odometry)
        self.time_sync = message_filters.ApproximateTimeSynchronizer([rgb_image_sub, depth_image_sub,cmd_vel_sub], 1, 0.05)
        self.time_sync.registerCallback(self.image_callback)

        self.frame_count = 0
        self.start = time.time()

    def image_callback(self,rgb_img_msg,depth_img_msg,odom_msg):
        
        if(abs(odom_msg.twist.twist.linear.x)>0.01 and abs(odom_msg.twist.twist.angular.z)>0.01):
            print("Only run hand detection model when the robot is not moving")
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_img_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_img_msg,"passthrough")
           
            depth_array = np.array(depth_image, dtype=np.float32)               # shape: (480, 640)
        except CvBridgeError as e:
            print(e)
            return 

        flag_result , gesture_result,confidence,c_x,c_y= self.detector.detect_image(cv_image)

        if flag_result:
        
            depth_handmidpoint = depth_array[c_x,c_y]

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
                    print("Detected gesture: {} with distance {}".format(gesture_result,depth_handmidpoint))

                    if(sum(self.gesture_recognition_history[key]) >= self.match_threshold):
                        self.checked_gesture = True         
                        self.last_gesture    = key
                        self.gesture_recognition_history = dict()

                        if(depth_handmidpoint<self.distance_threshold):
                            gesture_result_msg = Header()
                            gesture_result_msg.stamp = rospy.Time.now()
                            gesture_result_msg.frame_id = gesture_result
                            self.gesture_pub.publish(gesture_result_msg)
                        break

                    if not self.checked_gesture and gesture_result != self.last_gesture:
                        gesture_result = "Pending + {}".format(gesture_result)                    

        self.frame_count += 1
        # print('Average FPS: ', self.frame_count / (time.time() - self.start))

if __name__ == '__main__':
    rospy.init_node('hand_gesture_recognition', anonymous=True)

    ic = handtracker_webcam()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

