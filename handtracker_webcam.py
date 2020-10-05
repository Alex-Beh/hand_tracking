#
# Basic example of how to use the handtracker using a web camera as input.
#
# Author: Joakim Eriksson, joakim.eriksson@ri.se
#
#!/usr/bin/python3
import cv2
from hand_tracker import HandTracker
import numpy as np

import gesture_recognition

# so_file = "/home/alex-beh/code/Pose_folder/hand_tracking/gesture_recognition1.so"
# my_functions = ctypes.CDLL(so_file)

def draw_hand(frame, kp):
    # the points where we go back to the first midpoint of the hand
    clearpoints = [4,8,12,16,20]
    lk = None
    p = 0

    # print("Gesture: {}".format(gesture_recognition.gesture_recognition(kp.flatten())))
    gesture_result = gesture_recognition.gesture_recognition(kp.flatten())
    cv2.putText(frame, gesture_result , (100,100), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 

    # Draw the hand
    for keypoint in kp:
        if lk is not None:
            cv2.line(frame, (int(keypoint[0]),int(keypoint[1])),(int(lk[0]),int(lk[1])), (255,0,255), 2)
        lk = keypoint
        cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, (0,255,255), -1)

        if p in clearpoints:
            lk = kp[0]
        p = p + 1

    # kp_flat= kp.flatten()
    # cv2.putText(frame, '{} {}'.format(int(kp_flat[4]),int(kp_flat[5])), ((int(kp_flat[4]),int(kp_flat[5]))), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
    # cv2.putText(frame, '{} {}'.format(int(kp_flat[6]),int(kp_flat[7])), ((int(kp_flat[6]),int(kp_flat[7]))), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
    # cv2.putText(frame, '{} {}'.format(int(kp_flat[8]),int(kp_flat[9])), ((int(kp_flat[8]),int(kp_flat[9]))), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 

def draw_box(frame, box):
    # draw the box
    for i in range(0,4):
        cv2.line(frame, (int(box[i][0]),int(box[i][1])),(int(box[(i+1)&3][0]),int(box[(i+1)&3][1])), (255,255,255), 2)

if __name__ == '__main__':
    palm_model_path = "./models/palm_detection.tflite"
    landmark_model_path = "./models/hand_landmark.tflite"
    anchors_path = "./data/anchors.csv"

    cap = cv2.VideoCapture(0)

    detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                           box_shift=0.2, box_enlarge=1.3)

    # Capture video and do the hand-tracking
    while True:
        ret,frame = cap.read()
        image = frame[:,:,::-1]
        kp, box = detector(image)

        if kp is not None:
            draw_hand(frame, kp)
            # draw_box(frame, box)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
