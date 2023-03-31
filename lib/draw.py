import numpy as np
import cv2


def add_3d_detection(img, results, calib):
    dim = results[0:3]
    pos = results[3:6]
    ori = results[6:7]
    cl = results[9:10]
    pos[1] = pos[1] + dim[0] / 2
    # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
    # dim = dim / self.dim_scale
    # cl = self.names[cat]
    box_3d = compute_box_3d(dim, pos, ori)
    box_2d = project_to_image(box_3d, calib)
    img = draw_box_3d(img, box_2d, colors[int(cl)])
    return img


def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    pts_3d_homo = np.concatenate(
        [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    # import pdb; pdb.set_trace()
    return pts_2d


def compute_box_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, 0]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, 0]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)


def draw_box_3d(image, corners, c=(225, 238, 160)):
    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                     (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), c, 1, lineType=cv2.LINE_AA)
        # if ind_f == 0:
        #     cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
        #              (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
        #     cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
        #              (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
    return image


def draw_bev(bev, bev_labels):
    if bev is None:
        bev = np.zeros((500, 200, 3))

    for i in range(1, 7):
        cv2.circle(bev, (100, 400), i * 50, (150, 150, 150), 1)
    multiple = 6  # 放大倍数
    for bev_label in bev_labels:
        # [class, width, length, x, y, rotation]
        # 下面几个参数，可能需要根据自己的数据进行调整
        x = 100 + int(bev_label[3] * multiple)  # 矩形框的中心点x
        y = 400 - int(bev_label[4] * multiple)  # 矩形框的中心点y
        anglePi = bev_label[5]  # 矩形框的倾斜角度（长边相对于水平）
        width, height = int(bev_label[1] * multiple), int(bev_label[2] * multiple)  # 矩形框的宽和高
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


colors = [[173, 202, 25],
          [181, 199, 140],
          [225, 238, 160],
          [233, 231, 190],
          [199, 237, 190],
          [183, 213, 214],
          [116, 186, 209],
          [172, 206, 230]]