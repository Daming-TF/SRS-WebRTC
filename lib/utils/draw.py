import io
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np


class Drawer(object):
    def __init__(self, debug):
        self.canvas = None
        self.debug = debug
        self.canvas_shape = None

    def set_canvas(self, img_bgr):
        self.canvas = img_bgr.copy()
        self.canvas_shape = self.canvas.shape

    def get_canvas(self):
        return self.canvas

    def __call__(self, hand, pose_landmarks=None):
        if hand.landmark is not None:
            if self.debug:
                self.draw_rotated_rect(hand.rect_roi_coord)
                self.draw_text(hand.handness, hand.type, hand.rect, pos=hand.landmark[0][:2])

            self.draw_point(hand.landmark)
            self.draw_gesture(hand.landmark, hand.gesture_label)

        if self.debug and (hand.img_roi_bgr is not None):
            self.copy_past_roi(hand.type, hand.img_roi_bgr)

    def copy_past_roi(self, hand_type, img_roi, color=(0, 20, 229), thickness=2, offset=(5, 10),
                      font=cv2.FONT_HERSHEY_SIMPLEX):
        size = np.minimum(self.canvas_shape[0], self.canvas_shape[1])
        font_scale = size * 0.001

        img_roi = cv2.resize(img_roi, (int(size * 0.25), int(size * 0.25)), interpolation=cv2.INTER_LINEAR)
        h, w = img_roi.shape[0], img_roi.shape[1]
        text = hand_type.upper() + " Input"

        if hand_type == "right":
            self.canvas[-h:, :w] = img_roi
            pt1 = (0, self.canvas.shape[0] - h)
            pt2 = (w, self.canvas.shape[0])

        else:
            self.canvas[-h:, -w:] = img_roi
            pt1 = (self.canvas.shape[1] - w, self.canvas.shape[0] - h)
            pt2 = (self.canvas.shape[1], self.canvas.shape[0])

        self.canvas = cv2.rectangle(self.canvas, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
        pt3 = (pt1[0] + offset[0], pt1[1] - offset[1])
        cv2.putText(self.canvas, text, pt3, font, font_scale, color, thickness, cv2.LINE_AA)

    def draw_pose_landmark(self, pose_landmarks):
        if (pose_landmarks is not None) and self.debug:
            link_pairs = [
                [15, 13], [13, 11], [11, 5], [12, 14], [14, 16], [12, 6], [9, 7], [7, 5], [5, 6], [6, 8],
                [8, 10], [9, 17], [10, 18]]

            color = [
                (255., 0., 85.), (255., 0., 0.), (255., 85., 0.), (255., 170., 0.), (255., 255., 0.),
                (170., 255., 0.), (0., 170., 255.), (0., 85., 255.), (0., 0., 255.), (255., 0., 170.),
                (170., 0., 255.), (255., 0., 255.), (85., 0., 255.), (0., 125., 155.), (0., 85., 255.)]

            for j in range(pose_landmarks.shape[0]):
                for land_id in range(5, 19):
                    if pose_landmarks[j, land_id, 2] > 0.5:
                        center = (int(pose_landmarks[j, land_id, 0]), int(pose_landmarks[j, land_id, 1]))
                        cv2.circle(self.canvas, center=center, radius=5, color=(255, 255, 0), thickness=-1)

                for idx, link in enumerate(link_pairs):
                    if pose_landmarks[j, link[0], 2] > 0.5 and pose_landmarks[j, link[1], 2] > 0.5:
                        joint_x1 = pose_landmarks[j, link[0], 0]
                        joint_y1 = pose_landmarks[j, link[0], 1]
                        joint_x2 = pose_landmarks[j, link[1], 0]
                        joint_y2 = pose_landmarks[j, link[1], 1]

                        start_point = (int(joint_x1), int(joint_y1))
                        end_point = (int(joint_x2), int(joint_y2))
                        cv2.line(self.canvas, pt1=start_point, pt2=end_point, color=color[idx], thickness=2,
                                 lineType=cv2.LINE_AA)

    def draw_gesture(self, coords, gesture,
                     color=(211, 0, 148), thickness=3, font_scale=1.0, font=cv2.FONT_HERSHEY_SIMPLEX,
                     offset=(10, 40)):
        position = coords[0][:2] + offset
        if gesture is not None:
            cv2.putText(self.canvas, gesture, (int(position[0]), int(position[1])), font, font_scale, color,
                        thickness, cv2.LINE_AA)

    def draw_rotated_rect(self, rect_roi_coords, color=(0, 20, 229), thickness=5):
        pt1 = tuple(rect_roi_coords[0].astype(np.int))
        pt2 = tuple(rect_roi_coords[1].astype(np.int))
        pt3 = tuple(rect_roi_coords[2].astype(np.int))
        pt4 = tuple(rect_roi_coords[3].astype(np.int))

        cv2.line(self.canvas, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
        cv2.line(self.canvas, pt1=pt2, pt2=pt4, color=color, thickness=thickness)
        cv2.line(self.canvas, pt1=pt4, pt2=pt3, color=color, thickness=thickness)
        cv2.line(self.canvas, pt1=pt3, pt2=pt1, color=color, thickness=thickness)

    def draw_point(self, points):
        i = 0
        rootx, rooty = None, None
        prex, prey = None, None

        for point in points:
            x = int(point[0])
            y = int(point[1])

            if i == 0:
                rootx = x
                rooty = y
            if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
                prex = rootx
                prey = rooty

            if (i > 0) and (i <= 4):
                cv2.line(self.canvas, pt1=(prex, prey), pt2=(x, y), color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
                cv2.circle(self.canvas, center=(x, y), radius=5, color=(0, 0, 255), thickness=-1)
            if (i > 4) and (i <= 8):
                cv2.line(self.canvas, pt1=(prex, prey), pt2=(x, y), color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
                cv2.circle(self.canvas, center=(x, y), radius=5, color=(0, 255, 255), thickness=-1)
            if (i > 8) and (i <= 12):
                cv2.line(self.canvas, pt1=(prex, prey), pt2=(x, y), color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                cv2.circle(self.canvas, center=(x, y), radius=5, color=(0, 255, 0), thickness=-1)
            if (i > 12) and (i <= 16):
                cv2.line(self.canvas, pt1=(prex, prey), pt2=(x, y), color=(255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                cv2.circle(self.canvas, center=(x, y), radius=5, color=(255, 255, 0), thickness=-1)
            if (i > 16) and (i <= 20):
                cv2.line(self.canvas, pt1=(prex, prey), pt2=(x, y), color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                cv2.circle(self.canvas, center=(x, y), radius=5, color=(255, 0, 0), thickness=-1)

            if len(point) == 3:
                z_min = np.min(points[:, 2])
                z_max = np.max(points[:, 2])

                c_val = (1. - (point[2] - z_min) / (z_max - z_min + 1e-10)) * 255.
                c_val = int(np.clip(c_val, 0, 255))
                cv2.circle(self.canvas, center=(x, y), radius=7, color=(c_val, c_val, c_val), thickness=2)

            prex = x
            prey = y
            i = i + 1

    def draw_text(self, handness, hand_type, rect, pos,
                  color=(0, 20, 229), thickness=2, font_scale=0.7, font=cv2.FONT_HERSHEY_SIMPLEX, offset=(10, 70)):
        position = pos + offset

        handness_text = f'{handness * 100:.2f}%'
        leftright_text = f" {hand_type.upper()}"

        if rect is not None:
            rotation_radian = rect.rotation
            rotation_degree = rotation_radian * 360 / (2 * math.pi)
            rotate_text = f" {rotation_degree:.2f}"
        else:
            rotate_text = ""

        text = handness_text + leftright_text + rotate_text

        # putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)
        cv2.putText(self.canvas, text, (int(position[0]), int(position[1])), font, font_scale, color, thickness,
                    cv2.LINE_AA)


class Draw3dLandmarks(object):
    def __init__(self, frame_size):
        self.frame_size = frame_size   # (h, w)
        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.elev = 11
        self.count = 0
        self.interval = 3

        self.connections = [
            [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
            [1, 2], [2, 3], [3, 4],
            [5, 6], [6, 7], [7, 8],
            [9, 10], [10, 11], [11, 12],
            [13, 14], [14, 15], [15, 16],
            [17, 18], [18, 19], [19, 20]
        ]
        self.tips = np.array([1, 0, 0, 0, 0,
                              1, 1, 1,
                              0, 0, 0,
                              0, 0, 0,
                              0, 0, 0,
                              0, 0, 0], dtype=bool)

    def get_azim(self):
        x = self.count % 360

        if (x >= 0) and (x < 180):
            y = pow(-1, x // 180) * x
        elif (x >= 180) and (x < 360):
            y = pow(-1, x // 180) * (360 - x)
        else:
            y = 0

        return y

    def set_lims(self):
        self.ax.view_init(elev=self.elev, azim=self.get_azim())
        self.count += self.interval

        self.ax.set_xlim(-0.1, 0.1)
        self.ax.set_ylim(-0.1, 0.1)
        self.ax.set_zlim(-0.1, 0.1)

        self.ax.axes.xaxis.set_ticklabels([])
        self.ax.axes.yaxis.set_ticklabels([])
        self.ax.axes.zaxis.set_ticklabels([])

    def draw(self, points, lcolor='#ff0000', rcolor='#0000ff'):
        plt.cla()
        self.set_lims()

        for ind, (i, j) in enumerate(self.connections):
            x, y, z = [np.array([points[i, c], points[j, c]]) for c in range(3)]
            self.ax.plot(x, z, -y, lw=2, c=lcolor if self.tips[ind] else rcolor)     # draw lines
            self.ax.scatter(x, z, -y, color='#FF0088', s=20)  # draw points

    def __call__(self, img_show, points):
        self.draw(points)

        buf = io.BytesIO()
        self.fig.savefig(buf, format="jpg", dpi=120)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.resize(cv2.imdecode(img_arr, cv2.IMREAD_COLOR), (self.frame_size[0], self.frame_size[0]))
        img_show = np.hstack([img_show, img])

        return img_show


def draw_point(img, points):
    i = 0
    for point in points:
        x = int(point[0])
        y = int(point[1])

        if i == 0:
            rootx = x
            rooty = y
        if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
            prex = rootx
            prey = rooty

        if (i > 0) and (i <= 4):
            cv2.line(img, pt1=(prex, prey), pt2=(x, y), color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
            cv2.circle(img, center=(x, y), radius=5, color=(0, 0, 255), thickness=-1)
        if (i > 4) and (i <= 8):
            cv2.line(img, pt1=(prex, prey), pt2=(x, y), color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
            cv2.circle(img, center=(x, y), radius=5, color=(0, 255, 255), thickness=-1)
        if (i > 8) and (i <= 12):
            cv2.line(img, pt1=(prex, prey), pt2=(x, y), color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
            cv2.circle(img, center=(x, y), radius=5, color=(0, 255, 0), thickness=-1)
        if (i > 12) and (i <= 16):
            cv2.line(img, pt1=(prex, prey), pt2=(x, y), color=(255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
            cv2.circle(img, center=(x, y), radius=5, color=(255, 255, 0), thickness=-1)
        if (i > 16) and (i <= 20):
            cv2.line(img, pt1=(prex, prey), pt2=(x, y), color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
            cv2.circle(img, center=(x, y), radius=5, color=(255, 0, 0), thickness=-1)

        if len(point) == 3:
            z_min = np.min(points[:, 2])
            z_max = np.max(points[:, 2])

            c_val = (1. - (point[2] - z_min) / (z_max - z_min + 1e-10)) * 255.
            c_val = int(np.clip(c_val, 0, 255))
            cv2.circle(img, center=(x, y), radius=7, color=(c_val, c_val, c_val), thickness=2)

        prex = x
        prey = y
        i = i + 1

    return img


def draw_rectangle(img, roi_box, color=(0, 20, 229), thickness=3):
    if isinstance(roi_box, np.ndarray):
        pt1 = (int(roi_box[0][0]), int(roi_box[0][1]))
        pt2 = (int(roi_box[3][0]), int(roi_box[3][1]))
    elif isinstance(roi_box, list):
        pt1 = (int(roi_box[0][0]), int(roi_box[0][1]))
        pt2 = (int(roi_box[1][0]), int(roi_box[1][1]))
    else:
        raise Exception(" [!] Type of roi_box is not considered!")

    img_show = cv2.rectangle(img, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
    return img_show


def draw_text(img, handness, righthand_prop, rect, color=(0, 20, 229), thickness=2, font_scale=0.7,
              font=cv2.FONT_HERSHEY_SIMPLEX, offset=(10, 30)):
    handness_text = f'{handness * 100:.2f}%'

    if righthand_prop >= 0.5:
        leftright_text = " RightHand"
    else:
        leftright_text = " LeftHand"

    if rect is not None:
        rotation_radian = rect.rotation
        rotation_degree = rotation_radian * 360 / (2 * math.pi)
        rotate_text = f" {rotation_degree:.2f} degree"
    else:
        rotate_text = ""

    text = handness_text + leftright_text + rotate_text

    # putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)
    img_show = cv2.putText(img, text, offset, font, font_scale, color, thickness, cv2.LINE_AA)
    return img_show


def draw_gesture(img, coords, gesture, color=(255, 255, 255), thickness=3, font_scale=0.8, font=cv2.FONT_HERSHEY_SIMPLEX,
                 offset=(10, 30)):
    position = coords[0][:2] + offset
    if gesture is not None:
        img_show = cv2.putText(img, gesture, (int(position[0]), int(position[1])), font, font_scale, color, thickness,
                               cv2.LINE_AA)
    else:
        img_show = img
    return img_show


def draw_rotated_rect(img, rect_roi_coords, color=(239, 80, 0), thickness=5):
    pt1 = tuple(rect_roi_coords[0].astype(np.int))
    pt2 = tuple(rect_roi_coords[1].astype(np.int))
    pt3 = tuple(rect_roi_coords[2].astype(np.int))
    pt4 = tuple(rect_roi_coords[3].astype(np.int))

    img = cv2.line(img, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
    img = cv2.line(img, pt1=pt2, pt2=pt4, color=color, thickness=thickness)
    img = cv2.line(img, pt1=pt4, pt2=pt3, color=color, thickness=thickness)
    img = cv2.line(img, pt1=pt3, pt2=pt1, color=color, thickness=thickness)

    return img


def copy_past_roi(img_show, img_roi, color=(0, 20, 229), thickness=2, offset=(5, 10), font_scale=0.7,
                  font=cv2.FONT_HERSHEY_SIMPLEX):
    img_roi = cv2.resize(img_roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    h, w = img_roi.shape[0], img_roi.shape[1]
    img_show[-h:, :w] = img_roi

    pt1 = (0, img_show.shape[0] - h)
    pt2 = (w, img_show.shape[0])
    img_show = cv2.rectangle(img_show, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

    text = "Input"
    pt3 = (pt1[0] + offset[0], pt1[1] - offset[1])
    img_show = cv2.putText(img_show, text, pt3, font, font_scale, color, thickness, cv2.LINE_AA)

    return img_show
