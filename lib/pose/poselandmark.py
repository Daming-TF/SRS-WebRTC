import platform
import numpy as np

from lib.core.api.human_action_config import config
from lib.core.api.human_action_detector import HumanActionDetector


class PoseLandmark(object):
    def __init__(self):
        self.num_humans = 1
        self.num_joints = 19
        self.num_dims = 3
        self.LEFT_SHOULDER_INDEX = 5
        self.RIGHT_SHOULDER_INDEX = 6
        self.LEFT_WRIST_INDEX = 9
        self.RIGHT_WRIST_INDEX = 10
        self.LEFT_HAND_INDEX = 17
        self.RIGHT_HAND_INDEX = 18
        self.thres = 0.5
        self.landmark = None

        sysstr = platform.system()
        if sysstr == "Windows":
            lib_name = "huya_face.dll"
        else:
            raise Exception(" [!] Pose-Landmark tested on WINDOWS only!")

        self.detector = HumanActionDetector(lib_name, config.vidCreateConfig)
        self.detector.addSubModel(b"./lib/models/pose/hyai_pc_sdk_body_v1.0.2.model")
        self.detector.setParam(config.SDKParamType.HYAI_MOBILE_SDK_PARAM_FACELIMIT, 8)

    def __call__(self, img_bgr):
        pred_result = self.detector.run(img_bgr,
                                        config.PixelFormat.HYPixelFormatBGR,
                                        config.ImageRotate.ImageRotationCCW0,
                                        config.detectConfig)
        hand_boxes = self.post_process(pred_result, img_bgr.shape)
        self.convert_to_array(pred_result)
        return hand_boxes, self.landmark

    def convert_to_array(self, det_result):
        if det_result.human_count > 0:
            self.landmark = np.zeros((self.num_humans, self.num_joints, self.num_dims), dtype=np.float32)
            for j in range(self.num_humans):
                for land_id in range(self.num_joints):
                    self.landmark[j, land_id, :] = np.array([
                        det_result.d_humans[j].points_array[land_id].x,
                        det_result.d_humans[j].points_array[land_id].y,
                        det_result.d_humans[j].keypoints_score[land_id]], dtype=np.float32)
        else:
            self.landmark = None

    def post_process(self, det_result, img_shape):
        boxes = list()

        if det_result.human_count > 0:
            for j in range(self.num_humans):
                left_shoulder = np.array([
                    det_result.d_humans[j].points_array[self.LEFT_SHOULDER_INDEX].x,
                    det_result.d_humans[j].points_array[self.LEFT_SHOULDER_INDEX].y])
                right_shoulder = np.array([
                    det_result.d_humans[j].points_array[self.RIGHT_SHOULDER_INDEX].x,
                    det_result.d_humans[j].points_array[self.RIGHT_SHOULDER_INDEX].y])
                shoulder_len = np.sqrt(np.sum(np.power((left_shoulder - right_shoulder), 2)))

                for i, (hand_index, wrist_index) in enumerate(zip(
                        [self.LEFT_HAND_INDEX, self.RIGHT_HAND_INDEX],
                        [self.LEFT_WRIST_INDEX, self.RIGHT_WRIST_INDEX])):

                    if det_result.d_humans[j].keypoints_score[hand_index] > self.thres:
                        wrist = np.array([
                            det_result.d_humans[j].points_array[wrist_index].x,
                            det_result.d_humans[j].points_array[wrist_index].y])

                        # out of boundary
                        if (wrist[0] > img_shape[1]) or (wrist[1] > img_shape[0]):
                            break

                        hand = np.array([
                            det_result.d_humans[j].points_array[hand_index].x,
                            det_result.d_humans[j].points_array[hand_index].y])
                        hand_len = np.sqrt(np.sum(np.power((wrist - hand), 2)))
                        wh = np.maximum(hand_len * 1.4, shoulder_len * 0.5)

                        box = self.get_handbbox(hand, wh, img_shape)
                        if i == 0:
                            boxes.append({"type": "left", "box": box})
                        else:
                            boxes.append({"type": "right", "box": box})

        return boxes

    @staticmethod
    def get_handbbox(hand, wh, img_shape):
        h, w = img_shape[0], img_shape[1]

        bbox = np.array([[hand[0] - wh, hand[1] - wh],
                         [hand[0] + wh, hand[1] + wh]], dtype=np.float32)
        bbox[0][0] = np.maximum(0, bbox[0][0])
        bbox[0][1] = np.maximum(0, bbox[0][1])
        bbox[1][0] = np.minimum(bbox[1][0], w - 1)
        bbox[1][1] = np.minimum(bbox[1][1], h - 1)
        return bbox

    def release(self):
        self.detector.reset()
        self.detector.destroy()
