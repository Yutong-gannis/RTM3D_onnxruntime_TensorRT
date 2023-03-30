import cv2
import numpy as np
from lib.rtm3d_detector import RTMDetector


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

    cap = cv2.VideoCapture(r"test3_Trim.mp4")
    # cap = cv2.VideoCapture(r"G:\dataset\road_test\0009\0aa41b955831245e365e98cf9800c7b1.mp4")
    time_tol = 0
    num = 0
    while True:
        num += 1
        ret, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img = img[216:600, :, :]
        #img = img[100:612, 374:986, :]
        #img = cv2.resize(img, (1280, 384))
        ret = detector.run(img)
        results = ret['results']
        bboxes_3d = np.concatenate(
            [results[:, 3:6], results[:, 0:3], results[:, 6:7]],
            axis=1).astype(np.float32)
        bev_label = np.concatenate(
            [results[:, 8:9], results[:, 0:1], results[:, 2:3], results[:, 3:4], results[:, 5:6], results[:, 6:7]],
            axis=1).astype(np.float32)
        bev = draw_bev(None, bev_label)
        imgpts_list = make_imgpts_list(bboxes_3d, calib_numpy[:, :3])
        img_draw = draw_smoke_3d(img, imgpts_list)
        cv2.imshow("bev", bev)
        cv2.imshow('3d', img_draw)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        time_str = ''
        for stat in time_stats:
            time_tol = time_tol + ret[stat]
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        time_str = time_str + '{} {:.3f}s |'.format('tol', time_tol / num)
        print(time_str)


def draw_bev(bev, bev_labels):
    if bev is None:
        bev = np.zeros((500, 200, 3))

    for i in range(1, 7):
        cv2.circle(bev, (100, 400), i * 50, (10, 10, 10), 0)
    for bev_label in bev_labels:
        # [class, width, length, x, y, rotation]
        # 下面几个参数，可能需要根据自己的数据进行调整
        x = 100 + int(bev_label[3] * 5)  # 矩形框的中心点x
        y = 400 - int(bev_label[4] * 5)  # 矩形框的中心点y
        anglePi = bev_label[5]  # 矩形框的倾斜角度（长边相对于水平）
        width, height = int(bev_label[1] * 5), int(bev_label[2] * 5)  # 矩形框的宽和高
        cosA = np.sin(anglePi)
        sinA = -np.cos(anglePi)

        x1 = x - 0.5 * width
        y1 = y - 0.5 * height
        x0 = x + 0.5 * width
        y0 = y1
        x2 = x1
        y2 = y + 0.5 * height
        x3 = x0
        y3 = y2

        x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
        y0n = (x0 - x) * sinA + (y0 - y) * cosA + y
        x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
        y1n = (x1 - x) * sinA + (y1 - y) * cosA + y
        x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
        y2n = (x2 - x) * sinA + (y2 - y) * cosA + y
        x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
        y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

        # 根据得到的点，画出矩形框
        cv2.line(bev, (int(x0n), int(y0n)), (int(x1n), int(y1n)), (255, 255, 255), 1, 4)
        cv2.line(bev, (int(x1n), int(y1n)), (int(x2n), int(y2n)), (255, 255, 255), 1, 4)
        cv2.line(bev, (int(x2n), int(y2n)), (int(x3n), int(y3n)), (255, 255, 255), 1, 4)
        cv2.line(bev, (int(x0n), int(y0n)), (int(x3n), int(y3n)), (255, 255, 255), 1, 4)
    return bev


def make_imgpts_list(bboxes_3d, K):
    rvec = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tvec = np.array([[0.0], [0.0], [0.0]])
    imgpts_list = []
    for box3d in bboxes_3d:
        locs = np.array(box3d[0:3])
        rot_y = np.array(box3d[6])
        height, width, length = box3d[3:6]
        _, box2d, box3d = encode_label(K, rot_y, np.array([length, height, width]), locs)
        if np.all(box2d == 0):
            continue
        imgpts, _ = cv2.projectPoints(box3d.T, rvec, tvec, K, 0)
        imgpts_list.append(imgpts)
    return imgpts_list


def draw_smoke_3d(img, imgpts_list):
    connect_line_id = [[1, 0], [2, 7], [3, 6], [4, 5], [1, 2], [2, 3], [3, 4], [4, 1], [0, 7], [7, 6], [6, 5], [5, 0]]
    img_draw = img.copy()
    for imgpts in imgpts_list:
        for p in imgpts:
            p_x, p_y = int(p[0][0]), int(p[0][1])
            cv2.circle(img_draw, (p_x, p_y), 1, (0, 255, 0), -1)
        for i, line_id in enumerate(connect_line_id):
            p1 = (int(imgpts[line_id[0]][0][0]), int(imgpts[line_id[0]][0][1]))
            p2 = (int(imgpts[line_id[1]][0][0]), int(imgpts[line_id[1]][0][1]))
            if i <= 3:  # body
                color = (255, 0, 0)
            elif i <= 7:  # head
                color = (0, 0, 255)
            else:  # tail
                color = (255, 255, 0)
            color = (225, 238, 160)
            cv2.line(img_draw, p1, p2, color, 1)
    return img_draw


def encode_label(K, ry, dims, locs):
    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]

    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]

    x_corners += -np.float32(l) / 2
    y_corners += -np.float32(h)
    z_corners += -np.float32(w) / 2

    corners_3d = np.array([x_corners, y_corners, z_corners])
    rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = np.matmul(rot_mat, corners_3d)
    corners_3d += np.array([x, y, z]).reshape([3, 1])

    loc_center = np.array([x, y - h / 2, z])
    proj_point = np.matmul(K, loc_center)
    proj_point = proj_point[:2] / proj_point[2]

    corners_2d = np.matmul(K, corners_3d)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([
        min(corners_2d[0]),
        min(corners_2d[1]),
        max(corners_2d[0]),
        max(corners_2d[1])
    ])

    return proj_point, box2d, corners_3d

if __name__ == '__main__':
    demo()
