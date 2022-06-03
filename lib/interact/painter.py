from lib.hands.hands_tracker import HandsTracker
import os
import cv2
import numpy as np


class Mypainter:
    def __init__(self, args, image_path):
        self.over_layer_list = self.get_image(image_path)
        self.header = self.over_layer_list[0]
        self.color = [255, 0, 0]        # 默认颜色
        self.brush_thickness = 15
        self.eraser_thickness = 60
        self.img_canvas = np.zeros((720, 1280, 3), np.uint8)
        self.img_inv = np.zeros((720, 1280, 3), np.uint8)
        self.xp = 0
        self.yp = 0
        self.tipIds = [4, 8, 12, 16, 20]
        self.tracker = HandsTracker(capability=args.capability, roi_mode=args.roi_mode, debug=args.debug,
                                    show_time=args.show_time)

    def get_image(self, image_path):
        over_layer_list = []
        image_names = os.listdir(image_path)
        for image_name in image_names:
            image = cv2.imread(os.path.join(image_path, image_name))
            over_layer_list.append(image)
        return over_layer_list

    def draw_painter_track(self, img):
        imgGray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, self.img_canvas)

        img[0:1280][0:153] = self.header
        return img

    def fingersUp(self, landmarks):        # lmList表示[id, x, y]
        fingers = []
        if landmarks[self.tipIds[0]][0] > landmarks[self.tipIds[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if landmarks[self.tipIds[id]][1] < landmarks[self.tipIds[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def __call__(self, image):
        image, hand_list = self.tracker(image)
        for hand_info in hand_list:
            if hand_info.landmark is None:
                continue
            landmarks = hand_info.landmark
            x1, y1 = landmarks[8][:2]
            x1 = int(x1)
            y1 = int(y1)
            # x2, y2 = landmarks[8][:2]
            fingers = self.fingersUp(landmarks)

            if fingers[1] and fingers[2]:
                if y1 < 153:
                    if 0 < x1 < 320:
                        self.header = self.over_layer_list[0]
                        self.color = [50, 128, 250]
                    elif 320 < x1 < 640:
                        self.header = self.over_layer_list[1]
                        self.color = [0, 0, 255]
                    elif 640 < x1 < 960:
                        self.header = self.over_layer_list[2]
                        self.color = [0, 255, 0]
                    elif 960 < x1 < 1280:
                        self.header = self.over_layer_list[3]
                        self.color = [0, 0, 0]
            # image[0:1280][0:153] = self.header

            if fingers[1] and fingers[2] == False:  # Index finger is up
                cv2.circle(image, (x1, y1), 15, self.color, cv2.FILLED)     # mark the keys of index fingers
                print("Drawing Mode")
                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = x1, y1

                if self.color == [0, 0, 0]:
                    cv2.line(image, (self.xp, self.yp), (x1, y1), self.color, self.eraser_thickness)
                    cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1), self.color, self.eraser_thickness)
                else:
                    cv2.line(image, (self.xp, self.yp), (x1, y1), self.color, self.brush_thickness)
                    cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1), self.color, self.brush_thickness)

            self.xp, self.yp = x1, y1
        return self.draw_painter_track(image)







