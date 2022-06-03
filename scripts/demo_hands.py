import cv2
import os
import sys
from argparse import ArgumentParser
import subprocess
import threading
import queue

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from lib.hands.hands_tracker import HandsTracker


parser = ArgumentParser()
parser.add_argument("--camera", default=0, action="store_true", help="open camera, default is test video")
parser.add_argument("--debug", default=1, action="store_true", help="debug mode show bbox and some other information")

parser.add_argument("--roi_mode", type=int, default=0, choices=[0, 1],
                    help="0: hand-detector, 1: pose-landmark for providing the position of hands")
parser.add_argument("--capability", default=1, type=int, choices=[0, 1],
                    help="model capability, 1 for large and 0 for lite hand-models")

parser.add_argument("--inp", default=r"http://47.112.130.31:8080/live/livestream.flv")
parser.add_argument("--fps", default=30, type=int,
                    help='frame-per-second used in saving video when works on webcam mode')
parser.add_argument("--save_path", default=r"./saves/test.mp4")
parser.add_argument("--rtmp", default=r'rtmp://47.112.130.31/live/livestream/Ming')

opts = parser.parse_args()


def image_sample(q, cap):
    #抽帧压入q队列
    while True:
        q.put(cap.read()[1])
        # if q.qsize() > 20:
        #     q.get() #尽量确保实时性，仅保留最新的一帧


def image_push(q):
    # capability 0 for the light model and 1 for large model
    tracker = HandsTracker(capability=opts.capability, roi_mode=opts.roi_mode, debug=opts.debug)

    while 1:
        cap = cv2.VideoCapture(opts.inp)
        pipe = pipe_init(cap, opts.rtmp)
        while cap.isOpened():
            if not q.empty():
                frame = q.get()
            # success, frame = cap.read()

            else:
                print("Ignoring empty frame.")  # If loading a video, use 'break' instead of 'continue'.
                cap.release()
                pipe.terminate()
                break

            # frame = cv2.flip(frame, flipCode=1)  # Flip the image horizontally for a later selfie-view display.
            canvas = tracker(frame)
            # canvas = cv2.flip(canvas, 1)
            cv2.imshow("Window", canvas)
            pipe.stdin.write(canvas.tostring())

            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                pipe.terminate()
                break
        cap.release()
        pipe.terminate()

    # cap.release()
    # pipe.terminate()


def pipe_init(cap, rtmp):
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(fps, '\t', width, '\t', height)
    width = int(0.25 * width)
    height = int(0.25 * height)
    # sizeStr = str(size[0]) + 'x' + str(size[1])

    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', "{}x{}".format(width, height),
               '-r', str(25),
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-flvflags', 'no_duration_filesize',
               '-f', 'flv',
               rtmp]

    return subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)


def main():
    q = queue.Queue()
    # cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(opts.inp)

    threadList = [threading.Thread(target=image_sample, args=(q, cap)),
                  threading.Thread(target=image_push, args=(q,))]
    for t in threadList:
        t.start()


if __name__ == "__main__":
    main()
