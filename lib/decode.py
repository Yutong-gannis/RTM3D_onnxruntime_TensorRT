import torch
import torch.nn as nn
import numpy as np


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)

    topk_ys = (topk_inds.float() / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind.float() / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def gen_position(kps, dim, rot, meta, const):
    b = kps.size(0)
    c = kps.size(1)
    opinv = meta['trans_output_inv']
    calib = meta['calib']

    opinv = opinv.unsqueeze(1)
    opinv = opinv.expand(b, c, -1, -1).contiguous().view(-1, 2, 3).float()
    kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2)
    hom = torch.ones(b, c, 1, 9).cuda()
    kps = torch.cat((kps, hom), dim=2).view(-1, 3, 9)
    kps = torch.bmm(opinv, kps).view(b, c, 2, 9)
    kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # 16.32,18
    si = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1]
    alpha_idx = rot[:, :, 1] > rot[:, :, 5]
    alpha_idx = alpha_idx.float()
    alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)
    alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
    alpna_pre = alpna_pre.unsqueeze(2)

    rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si)
    rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
    rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi

    calib = calib.unsqueeze(1)
    calib = calib.expand(b, c, -1, -1).contiguous()
    kpoint = kps[:, :, :16]
    f = calib[:, :, 0, 0].unsqueeze(2)
    f = f.expand_as(kpoint)
    cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)
    cxy = torch.cat((cx, cy), dim=2)
    cxy = cxy.repeat(1, 1, 8)  # b,c,16
    kp_norm = (kpoint - cxy) / f

    l = dim[:, :, 2:3]
    h = dim[:, :, 0:1]
    w = dim[:, :, 1:2]
    cosori = torch.cos(rot_y)
    sinori = torch.sin(rot_y)

    B = torch.zeros_like(kpoint)
    C = torch.zeros_like(kpoint)

    kp = kp_norm.unsqueeze(3)  # b,c,16,1
    const = const.expand(b, c, -1, -1)
    A = torch.cat([const, kp], dim=3)

    B[:, :, 0:1] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 1:2] = h * 0.5
    B[:, :, 2:3] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 3:4] = h * 0.5
    B[:, :, 4:5] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 5:6] = h * 0.5
    B[:, :, 6:7] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 7:8] = h * 0.5
    B[:, :, 8:9] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 9:10] = -h * 0.5
    B[:, :, 10:11] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 11:12] = -h * 0.5
    B[:, :, 12:13] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 13:14] = -h * 0.5
    B[:, :, 14:15] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 15:16] = -h * 0.5

    C[:, :, 0:1] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 1:2] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 2:3] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 3:4] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 4:5] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 5:6] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 6:7] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 7:8] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 8:9] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 9:10] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 10:11] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 11:12] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 12:13] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 13:14] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 14:15] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 15:16] = l * 0.5 * sinori + w * 0.5 * cosori

    B = B - kp_norm * C

    AT = A.permute(0, 1, 3, 2)
    AT = AT.view(b * c, 3, 16)
    A = A.view(b * c, 16, 3)
    B = B.view(b * c, 16, 1).float()

    pinv = torch.bmm(AT, A)
    pinv = torch.inverse(pinv)  # b*c 3 3

    pinv = torch.bmm(pinv, AT)
    pinv = torch.bmm(pinv, B)
    pinv = pinv.view(b, c, 3, 1).squeeze(3)

    return pinv, rot_y, kps


def car_pose_decode_faster(heat, kps, dim, rot, prob, K=100, thresh=0.3, meta=None, const=None):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    clses = clses.view(batch, K, 1).float()
    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)

    scores = scores.view(batch, K, 1)

    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 8)
    prob = _transpose_and_gather_feat(prob, inds)[:, :, 0]
    prob = prob.view(batch, K, 1)
    position, rot_y, kps_inv = gen_position(kps, dim, rot, meta, const)
    bboxes_kp = kps.view(kps.size(0), kps.size(1), 9, 2)
    box_min, _ = torch.min(bboxes_kp, dim=2)
    box_max, _ = torch.max(bboxes_kp, dim=2)
    bboxes = torch.cat((box_min, box_max), dim=2)
    hm_score = kps[:, :, 0:9]
    # detections = torch.cat([bboxes, scores, kps_inv, dim, hm_score, rot_y, position, prob, clses], dim=2)
    detections = torch.cat([dim, position, rot_y, scores, clses], dim=2)[0]
    detections = detections[torch.where(detections[:, -2] > thresh)[0], :].detach().cpu().numpy()
    return detections
