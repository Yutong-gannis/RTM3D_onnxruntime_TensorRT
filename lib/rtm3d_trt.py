import time
import numpy as np
import torch
import cv2
from lib.decode import car_pose_decode_faster
from lib.image import get_affine_transform
from lib.AB3DMOT_libs.model import AB3DMOT
from lib.engine import Engine
from lib.draw import add_3d_detection, draw_bev


class RTMDetector:
    def __init__(self, model_path, calib_path):
        self.batch = 1
        self.device = torch.device('cuda:0')
        self.engine = Engine(model_path)

        self.input_h = 384
        self.input_w = 1280
        self.down_ratio = 4
        self.scale = 1

        self.thresh = 0.45
        self.K = 30  # 最大目标数
        const = torch.Tensor(
            [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
             [-1, 0], [0, -1], [-1, 0], [0, -1]])
        self.const = const.unsqueeze(0).unsqueeze(0)
        self.const = self.const.to(self.device)

        self.tracker = AB3DMOT()

        calib_file = open(calib_path, 'r')
        for i, line in enumerate(calib_file):
            if i == 2:
                self.calib_np = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                self.calib_np = self.calib_np.reshape(3, 4)
                break

        c = np.array([self.input_w / 2., self.input_h / 2.], dtype=np.float32)
        s = max(self.input_h, self.input_w) * 1.0
        meta = {'c': c, 's': s,
                'out_height': self.input_h // self.down_ratio,
                'out_width': self.input_w // self.down_ratio}
        trans_output_inv = get_affine_transform(c, s, 0, [meta['out_width'], meta['out_height']], inv=1)
        trans_output_inv = torch.from_numpy(trans_output_inv)
        trans_output_inv = trans_output_inv.unsqueeze(0)
        meta['trans_output_inv'] = trans_output_inv.to(self.device)

        calib = torch.from_numpy(self.calib_np).unsqueeze(0).to(self.device)
        meta['calib'] = calib
        self.meta = meta

    def pre_process(self, img):
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img)
        return img

    def run(self, img, im1):
        img = self.pre_process(img)
        hm, hps, rot, dim, prob = self.engine(img)
        results = car_pose_decode_faster(hm, hps, dim, rot, prob, K=self.K, thresh=self.thresh, meta=self.meta, const=self.const)
        dets = results[:, :7]
        info = results[:, 7:9]  # 类别
        dets_all = {'dets': dets, 'info': info}
        results = self.tracker.update(dets_all)
        show_img = self.draw_result(im1, results)
        return results, show_img

    def draw_result(self, img, results):
        bev_label = np.concatenate(
            [results[:, 8:9], results[:, 0:1], results[:, 2:3], results[:, 3:4], results[:, 5:6], results[:, 6:7]],
            axis=1).astype(np.float32)
        bev = draw_bev(None, bev_label)
        for bbox in results:
            img = add_3d_detection(img, bbox, self.calib_np)
        show_img = np.hstack((np.uint8(cv2.resize(img, (1333, 400))), np.uint8(cv2.resize(bev, (160, 400)))))
        return show_img
