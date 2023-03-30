import cv2
import numpy as np
import time
import torch
from lib.decode import car_pose_decode_faster
from lib.image import get_affine_transform
from lib.msra_resnet import get_pose_net
from lib.AB3DMOT_libs.model import AB3DMOT


def create_model(heads, head_conv):
    model = get_pose_net(num_layers=18, heads=heads, head_conv=head_conv)
    return model


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                state_dict[k] = model_state_dict[k]
    for k in model_state_dict:
        if not (k in state_dict):
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    return model


class RTMDetector:
    def __init__(self, model_path, calib_numpy):
        self.device = torch.device('cuda:0')
        self.flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8]]
        self.num_classes = 8  # 检测类别数
        self.heads = {'hm': self.num_classes, 'hps': 18, 'rot': 8, 'dim': 3, 'prob': 1}  # 检测头
        self.head_conv = 64
        self.model = create_model(self.heads, self.head_conv)
        self.model = load_model(self.model, model_path)
        self.model = self.model.to(self.device).eval()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.input_h = 384
        self.input_w = 1280
        self.down_ratio = 4
        self.scale = 1

        self.thresh = 0.5
        self.K = 30  # 最大目标数
        self.nms = False
        const = torch.Tensor(
            [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
             [-1, 0], [0, -1], [-1, 0], [0, -1]])
        self.const = const.unsqueeze(0).unsqueeze(0)
        self.const = self.const.to(self.device)

        self.tracker = AB3DMOT()

        c = np.array([self.input_w / 2., self.input_h / 2.], dtype=np.float32)
        s = max(self.input_h, self.input_w) * 1.0
        meta = {'c': c, 's': s,
                'out_height': self.input_h // self.down_ratio,
                'out_width': self.input_w // self.down_ratio}
        trans_output_inv = get_affine_transform(c, s, 0, [meta['out_width'], meta['out_height']], inv=1)
        trans_output_inv = torch.from_numpy(trans_output_inv)
        trans_output_inv = trans_output_inv.unsqueeze(0)
        meta['trans_output_inv'] = trans_output_inv.to(self.device)

        calib = torch.from_numpy(calib_numpy).unsqueeze(0).to(self.device)
        meta['calib'] = calib
        self.meta = meta

    def pre_process(self, image):
        image = ((image / 255. - self.mean) / self.std).astype(np.float32)
        images = image.transpose(2, 0, 1).reshape(1, 3, self.input_h, self.input_w)
        images = torch.from_numpy(images)
        return images

    def process(self, images, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = car_pose_decode_faster(output['hm'], output['hps'], output['dim'], output['rot'],
                                          prob=output['prob'], K=self.K, meta=self.meta, const=self.const)
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        score = dets[0, :, 4:5]
        dim = dets[0, :, 23:26]
        rot_y = dets[0, :, 35:36]
        position = dets[0, :, 36:39]
        cat = dets[0, :, 40:41]
        results = np.concatenate([dim, position, rot_y, score, cat], axis=1)
        results = results[np.where(results[:, -2] > self.thresh), :][0]
        # bbox score kps kps_score dim rot_y position prob
        # 0:4  4:5   5:23 23:32    32:35 35:36 36:39 39:40
        # dim position  rot_y score class
        # 0:3      3:6    6:7   7:8   8:9
        return results

    def run(self, image):
        pre_time, dec_time, post_time, track_time, tot_time = 0, 0, 0, 0, 0

        start_time = time.time()
        images = self.pre_process(image)
        images = images.to(self.device)
        pre_process_time = time.time()
        torch.cuda.synchronize()
        pre_time += pre_process_time - start_time

        output, dets, forward_time = self.process(images, return_time=True)
        torch.cuda.synchronize()
        dec_time += forward_time - pre_process_time

        results = self.post_process(dets)
        torch.cuda.synchronize()
        post_process_time = time.time()
        post_time += post_process_time - forward_time

        dets = results[:, :7]
        info = results[:, 7:9]  # 类别
        dets_all = {'dets': dets, 'info': info}
        results = self.tracker.update(dets_all)
        end_time = time.time()
        track_time = end_time - post_process_time

        tot_time += end_time - start_time
        return {'results': results, 'tot': tot_time, 'pre': pre_time, 'dec': dec_time, 'post': post_time,
                'track': track_time}



