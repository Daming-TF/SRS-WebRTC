import time

from lib.hands.detector import HandDetModel
from lib.hands.hands import HandInfo, HandLandModel
from lib.pose import PoseLandmark
from lib.utils.draw import Drawer


class HandsTracker(object):
    def __init__(self, capability=1, threshold=0.5, roi_mode=0, debug=False):
        self.drawer = Drawer(debug)

        self.landmark_thres = threshold
        self.roi_mode = roi_mode

        self.left_hand = HandInfo('left')
        self.right_hand = HandInfo('right')

        if self.roi_mode == 0:
            self.detector = HandDetModel()      # hand detector
        elif self.roi_mode == 1:
            self.detector = PoseLandmark()      # pose landmark
        else:
            raise Exception(" [!] ROI mode only support 0 or 1!")

        self.name = "TFLite-Full" if capability > 0 else "TFLite-Lite"
        self.hand_model = HandLandModel(capability, self.roi_mode)    # using mediapipe's rotated rectangled roi logic

    def __call__(self, img_bgr):
        self.drawer.set_canvas(img_bgr)

        boxes = []
        pose_landmark = None
        if (self.left_hand.landmark is None) or (self.right_hand.landmark is None):
            start = time.time()
            boxes, pose_landmark = self.detector(img_bgr)
            end = time.time()
            print(f' [!] Detector time: {(end - start) * 1000:.2f} ms. - '
                  f'{"Hand-detector" if self.roi_mode == 0 else "Pose-landmarker"} model')

        start = time.time()
        self.hand_model.run_with_boxes(img_bgr, boxes, self.right_hand, self.left_hand)
        end = time.time()
        print(f' [!] Landmark time: {(end - start) * 1000:.2f} ms. - {self.name} model')

        for hand in [self.right_hand, self.left_hand]:
            self.drawer(hand)
        self.drawer.draw_pose_landmark(pose_landmark)

        return self.drawer.get_canvas()
