#
# Basic example of how to use the handtracker using a web camera as input.
#
# Author: Joakim Eriksson, joakim.eriksson@ri.se
#
#!/usr/bin/python3
import cv2
import numpy as np
import collections

from hand_tracker import HandTracker
import gesture_recognition

class handtracker_webcam:
    def __init__(self):
        print("Initialize handtracker_webcam")
        self.collection_length = 10
        self.match_threshold   = 7
        self.gesture_recognition_history = dict()

        self.checked_gesture = False
        self.last_gesture    = None

    def draw_hand(self,frame, kp):
        # the points where we go back to the first midpoint of the hand
        clearpoints = [4,8,12,16,20]
        lk = None
        p = 0

        # print("Gesture: {}".format(gesture_recognition.gesture_recognition(kp.flatten())))
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

    def draw_box(self,frame, box):
        # draw the box
        for i in range(0,4):
            cv2.line(frame, (int(box[i][0]),int(box[i][1])),(int(box[(i+1)&3][0]),int(box[(i+1)&3][1])), (255,255,255), 2)

if __name__ == '__main__':
    palm_model_path = "./../models/palm_detection.tflite"
    landmark_model_path = "./../models/hand_landmark.tflite"
    anchors_path = "./../data/anchors.csv"

    cap = cv2.VideoCapture(0)

    detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                           box_shift=0.2, box_enlarge=1.3)

    GestureRecognition = handtracker_webcam()
    # Capture video and do the hand-tracking
    while True:
        ret,frame = cap.read()
        image = frame[:,:,::-1]
        kp, box = detector(image)

        if kp is not None:
            GestureRecognition.draw_hand(frame, kp)
            # GestureRecognition.draw_box(frame, box)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
