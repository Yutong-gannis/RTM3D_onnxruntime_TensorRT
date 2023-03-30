import numpy as np
import cv2
import lib.kitti_read as kitti_utils


def Space2Bev(P0, side_range=(-20, 20),
              fwd_range=(0, 70),
              res=0.1):
    x_img = (P0[0] / res).astype(np.int32)
    y_img = (-P0[2] / res).astype(np.int32)

    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.floor(fwd_range[1] / res)) - 1

    return np.array([x_img, y_img])


def vis_create_bev(width=750, side_range=(-20, 20), fwd_range=(0, 70),
                   min_height=-2.5, max_height=1.5):
    ''' Project pointcloud to bev image for simply visualization

        Inputs:
            pointcloud:     3 x N in camera 2 frame
        Return:
            cv color image

    '''
    res = float(fwd_range[1] - fwd_range[0]) / width
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[:, :] = 255
    im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im_rgb


def vis_box_in_bev(im_bev, pos, dims, orien, width=750, gt=False, score=None,
                   side_range=(-20, 20), fwd_range=(0, 70),
                   min_height=-2.73, max_height=1.27):
    """ Project 3D bounding box to bev image for simply visualization
        It should use consistent width and side/fwd range input with
        the function: vis_lidar_in_bev

        Inputs:
            im_bev:         cv image
            pos, dim, orien: params of the 3D bounding box
        Return:
            cv color image

    """
    dim = dims.copy()
    res = float(fwd_range[1] - fwd_range[0]) / width

    R = kitti_utils.E2R(orien, 0, 0)
    pts3_c_o = [pos + R.dot(np.array([dim[0] / 2., 0, dim[2] / 2.0]).T),
                pos + R.dot(np.array([dim[0] / 2, 0, -dim[2] / 2.0]).T),
                pos + R.dot(np.array([-dim[0] / 2, 0, -dim[2] / 2.0]).T),
                pos + R.dot(np.array([-dim[0] / 2, 0, dim[2] / 2.0]).T), pos + R.dot([dim[0] / 1.5, 0, 0])]

    pts2_bev = []
    for index in range(5):
        pts2_bev.append(Space2Bev(pts3_c_o[index], side_range=side_range,
                                  fwd_range=fwd_range, res=res))

    if gt is False:
        lineColor3d = (100, 100, 0)
    else:
        lineColor3d = (0, 0, 255)
    if gt == 'next':
        lineColor3d = (255, 0, 0)
    if gt == 'g':
        lineColor3d = (0, 255, 0)
    if gt == 'b':
        lineColor3d = (255, 0, 0)
    if gt == 'n':
        lineColor3d = (5, 100, 100)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[1][0], pts2_bev[1][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[2][0], pts2_bev[2][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 1)

    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 1)

    if score is not None:
        show_text(im_bev, pts2_bev[4], score)
    return im_bev


def show_text(img, cor, score):
    txt = '{:.2f}'.format(score)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, txt, (cor[0], cor[1]),
                font, 0.3, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img


def draw_Box3D(box3d, ax, color_set='b'):
    for box in box3d:
        bbox = box.copy()
        box[0] = bbox[2]
        box[1] = bbox[0]
        box[2] = bbox[1]

    line_order = ([0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                  [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3])

    for k in line_order:
        ax.plot3D(*zip(box3d[k[0]].T, box3d[k[1]].T), lw=1.5, color=color_set)
