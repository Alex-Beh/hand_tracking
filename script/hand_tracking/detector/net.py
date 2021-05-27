#!/usr/bin/python3
import cv2
import numpy as np
from math import sqrt
import torch
import sys
from operator import itemgetter
from skimage import feature
import gesture_recognition

import functools
import operator

class HandKeypoint:
    def __init__(self):
        target = sys.argv[1] if len(sys.argv) > 1 else None
        self.model = torch.jit.load('/home/followme/followme_ws/src/hand_tracking/models/hand.pts', map_location=torch.device('cuda'))
        print('target... ', target)
        print('loading model... done')

        self.TRAIN_IMAGE_HEIGHT = self.TRAIN_IMAGE_WIDTH = 256
        self.LABEL_MIN = 0.3
        self.LABEL_HAND_MIN = 0.2
        # 손을 필수록 total과 dis가 비슷하고, 손을 접을수록 dis가 줄어드므로 ratio가 커진다.
        self.HAND_COLORS = [
            (100, 100, 100), (100, 0, 0), (150, 0, 0), (200, 0, 0), (255, 0, 0), (100, 100, 0), (150, 150, 0),
            (200, 200, 0), (255, 255, 0), (0, 100, 50), (0, 150, 75), (0, 200, 100), (0, 255, 125),
            (0, 50, 100), (0, 75, 150),(0, 100, 200),(0, 125, 255),(100, 0, 100),(150, 0, 150),(200, 0, 200), (255, 0, 255)
        ]
        _min = _max = None
            
    def nmslocation(self,src, threshold):
        locations = []
        blockwidth = 2
        rows, cols = src.shape

        arr = feature.peak_local_max(src, min_distance=2, threshold_abs=threshold, exclude_border=True, indices=True)
        new_arr = [(src[x][y], (x, y)) for x, y in arr]
        new_arr = sorted(new_arr, key=itemgetter(0), reverse=True)
        return new_arr
        
    def transform_net_input(self,tensor, source_img, hand_rect=None, tensor_idx=0):
        #             hand_rect.append((l_t[1], l_t[0], r_b[1], r_b[0], pos_x, pos_y))
        img = source_img.copy()
        if hand_rect is not None:
            l, t, r, b, _, _ = hand_rect[tensor_idx]
            img = img[t:b,l:r]
        rows, cols = len(img), len(img[0])
        ratio = min(tensor.shape[2] / rows, tensor.shape[3] / cols)
        mat = np.array([[ratio, 0, 0], [0, ratio, 0]])

        dst = cv2.warpAffine(img, mat, (tensor.shape[3], tensor.shape[2]))

        dst = dst / 255 - 0.5
        r, g, b = cv2.split(dst)

        tensor[tensor_idx][0] = torch.tensor(r, device=torch.device('cuda')).float()
        tensor[tensor_idx][1] = torch.tensor(g, device=torch.device('cuda')).float()
        tensor[tensor_idx][2] = torch.tensor(b, device=torch.device('cuda')).float()
        return ratio

    def detect_bbox(self,input_image):
        tensor = torch.zeros([1, 3, self.TRAIN_IMAGE_HEIGHT, self.TRAIN_IMAGE_WIDTH], device=torch.device('cuda'))
        rows, cols, _ = input_image.shape
        ratio_input_to_net = self.transform_net_input(tensor, input_image)
        heatmap = self.model.forward(tensor)[3]
        ratio_net_downsample = self.TRAIN_IMAGE_HEIGHT / heatmap.shape[2]
        rect_map_idx = heatmap.shape[1] - 3

        rectmap = []
        # copy three channel rect map
        for i in range(3):
            rectmap.append(np.copy(heatmap[0][i+rect_map_idx].cpu().detach().numpy()))
        canv = np.copy(rectmap[0])
        locations = self.nmslocation(rectmap[0], self.LABEL_MIN)
        hand_rect = []
        for loc_val, points in locations:
            pos_x, pos_y = points
            ratio_width = ratio_height = pixelcount = 0
            for m in range(max(pos_x-2, 0), min(pos_x+3, int(heatmap.shape[2]))):
                for n in range(max(pos_y-2, 0), min(pos_y+3, int(heatmap.shape[3]))):
                    ratio_width += rectmap[1][m][n]
                    ratio_height += rectmap[2][m][n]
                    pixelcount += 1

            if pixelcount > 0:
                ratio_width = min(max(ratio_width / pixelcount, 0), 1)
                ratio_height = min(max(ratio_height / pixelcount, 0), 1)
                ratio = ratio_net_downsample / ratio_input_to_net
                pos_x *= ratio
                pos_y *= ratio
                rect_w = ratio_width * self.TRAIN_IMAGE_WIDTH / ratio_input_to_net
                rect_h = ratio_height * self.TRAIN_IMAGE_HEIGHT / ratio_input_to_net

                l_t = (max(int(pos_x - rect_h/2), 0), max(int(pos_y - rect_w/2), 0))
                r_b = (min(int(pos_x + rect_h/2), rows - 1), min(int(pos_y + rect_w/2), cols - 1))

                hand_rect.append((l_t[1], l_t[0], r_b[1], r_b[0], pos_x, pos_y))

        return hand_rect

    def detect_hand(self,input_image, hand_rect):
        many_points = [None]*len(hand_rect)

        tensor = torch.zeros([len(hand_rect), 3, self.TRAIN_IMAGE_HEIGHT, self.TRAIN_IMAGE_WIDTH], device=torch.device('cuda'))
        ratio_input_to_net = [None]*len(hand_rect)
        for i in range(len(hand_rect)):
            ratio_input_to_net[i] = self.transform_net_input(tensor, input_image, hand_rect, i)

        net_result = self.model.forward(tensor)[3]
        ratio_net_downsample = self.TRAIN_IMAGE_HEIGHT / net_result.size()[2]
        heatmaps = []*len(hand_rect)
        many_points = []
        for rect_idx in range(len(hand_rect)):
            total_points = [[] for i in range(21)]
            x, y, _, _, _, _ = hand_rect[rect_idx]
            ratio = ratio_net_downsample / ratio_input_to_net[rect_idx]
            for i in range(21):
                heatmap = net_result[rect_idx][i].cpu().detach().numpy()
                points = self.nmslocation(heatmap, self.LABEL_HAND_MIN)
                if len(points):
                    _, point = points[0]
                    total_points[i] = (int(point[1]*ratio)+x, int(point[0]*ratio)+y)
            many_points.append(total_points)
        return many_points
    def pyramid_inference(self,input_image):

        rows, cols, _ = input_image.shape
        hand_rects = self.detect_bbox(input_image)
        if len(hand_rects) == 0:
            return [], []

        many_keypoints = self.detect_hand(input_image, hand_rects)
        for i in range(len(hand_rects)-1, -1, -1):
            missing_points = 0
            for j in range(21):
                if len(many_keypoints[i][j]) != 2:
                    missing_points += 1
            if missing_points > 5:
                hand_rects.pop(i)
                many_keypoints.pop(i)

        return many_keypoints, hand_rects

    def feed_frame(self,frame):
        many_keypoints, hand_rect = self.pyramid_inference(frame)

        for rect_idx, points in enumerate(many_keypoints):
            gesture_result = "Pending"
            # points_ = np.asarray(points)

            point_ = np.array([])
            valid_input = True
            for i in points:
                if i == []:
                    valid_input = False
                    continue
                point_ = np.append(point_,int(i[0]))
                point_ = np.append(point_,int(i[1]))

            if(valid_input):
                gesture_result = gesture_recognition.gesture_recognition(point_.flatten())
            print(gesture_result,len(points))
            rect = hand_rect[rect_idx]
            frame = cv2.rectangle(frame, rect[0:2], rect[2:4], (0, 0, 255), 6)

            point = (int(rect[5]), int(rect[4]))
            red = (0,0,255)
            green = (0,255,0)

            missing = 0
            if rect is None:
                continue
            for i, point in enumerate(points):
                if point is None or len(point) == 0:
                    missing+=1
                    continue
                frame = cv2.circle(frame, point, 6, self.HAND_COLORS[i], 6)
            print(points)
            frame = cv2.circle(frame, (points[0][0],points[0][1]), 6, (255,0,0), 6)

            per = f'{int(2100-missing*100)//21}% '
            for i in range(5):
                for j in range(3):
                    cnt = j+i*4+1
                    if len(points[cnt]) != 0 and len(points[cnt+1])!=0 :
                        frame = cv2.line(frame, points[cnt], points[cnt+1], (0, 255, 0), 2)

    #        per += ' ' + ','.join([str(int(x*100)/100) for x in state])
            text_pos = hand_rect[rect_idx][0:2]
            text_pos = (text_pos[0], text_pos[1]+5)
            frame = cv2.putText(frame, per, text_pos, 1, 3, (0, 0, 255), 3)
        frame = cv2.resize(frame, (512,512))
        if True:
            cv2.imshow('show', frame)
            # cv2.waitKey(0)
        return frame


    
def main():
    cap = cv2.VideoCapture(-1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    detector=HandKeypoint()
    while True:
        _, frame = cap.read()
        # cv2.imshow("Test",frame)
        # cv2.waitKey(0)
        if frame is None:
            break
        detector.feed_frame(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
if __name__ == '__main__':
    main()
