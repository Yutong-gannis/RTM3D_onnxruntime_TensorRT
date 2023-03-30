import numpy as np
import cv2
from lib.rtm3d_detector import RTMDetector
from lib.draw import add_3d_detection, draw_bev


def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib


def demo():
    time_stats = ['pre', 'dec', 'post', 'track']
    model_path = r"weights/kitti/model_250.pth"

    calib_path = r"calib/calib_kitti.txt"
    calib_numpy = read_clib(calib_path)
    detector = RTMDetector(model_path, calib_numpy)

    cap = cv2.VideoCapture(r"test1.mp4")
    # cap = cv2.VideoCapture(r"G:\dataset\road_test\0009\0aa41b955831245e365e98cf9800c7b1.mp4")
    time_tol = 0
    num = 0
    while True:
        num += 1
        ret, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img = img[126:510, :, :]
        # img = img[100:612, 374:986, :]
        # img = cv2.resize(img, (1280, 384))
        ret = detector.run(img)
        results = ret['results']
        bev_label = np.concatenate(
            [results[:, 8:9], results[:, 0:1], results[:, 2:3], results[:, 3:4], results[:, 5:6], results[:, 6:7]],
            axis=1).astype(np.float32)
        bev = draw_bev(None, bev_label)
        for bbox in results:
            img = add_3d_detection(img, bbox, calib_numpy)
        cv2.imshow("bev", bev)
        cv2.imshow('3d', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        time_str = ''
        for stat in time_stats:
            time_tol = time_tol + ret[stat]
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        time_str = time_str + '{} {:.3f}s |'.format('tol', time_tol / num)
        print(time_str)


if __name__ == '__main__':
    demo()
