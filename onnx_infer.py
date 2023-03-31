import time

import numpy as np
import cv2
from lib.rtm3d_ort import RTMDetector


def demo():
    model_path = r"weights/kitti/model_250.onnx"

    calib_path = r"calib/calib_kitti.txt"
    detector = RTMDetector(model_path, calib_path)

    cap = cv2.VideoCapture(r"test3_Trim.mp4")
    while True:
        start = time.time()
        ret, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img = img[176:560, :, :]
        im1 = img.copy()
        results, show_img = detector.run(img, im1)

        cv2.imshow('result', show_img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        end = time.time()
        print('infer time:', str(end - start))


if __name__ == '__main__':
    demo()
