import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.sintel_submission import get_cfg
from core.utils.misc import process_cfg
import datasets
import imageio
from utils import flow_viz
from utils import frame_utils

# from FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer
from raft import RAFT

from utils.utils import forward_interpolate
import itertools

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'kitti432':
            self._pad = [0, 0, 0, 432 - self.ht]
        elif mode == "kitti400":
            self._pad = [0, 0, 0, 400 - self.ht]
        elif mode == "kitti376":
            self._pad = [0, 0, 0, 376 - self.ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

TRAIN_SIZE = [288, 960]

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
  ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  #hs = [0]
  ws[-1] = image_shape[1] - patch_size[1]
  #ws.append((image_shape[1] - patch_size[1])//2)
  return [(h, w) for h in hs for w in ws]

import math
def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5 
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights


@torch.no_grad()
def eval_kitti_training_288960(model, iters=24, sigma=0.5):
    print("[eval_kitti_training_288960]")
    IMAGE_SIZE = [376, 1242]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    # test_dataset = datasets.KITTI(split='testing', aug_params=None)
    val_dataset = datasets.KITTI(split='training')
    print(hws)
    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        new_shape = image1.shape[1:]
        #print(new_shape)
        if new_shape[1] != IMAGE_SIZE[1] or new_shape[0] != IMAGE_SIZE[0]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[1] = new_shape[1]
            IMAGE_SIZE[0] = new_shape[0]
            hws = compute_grid_indices(IMAGE_SIZE)
            #print(hws)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        # image1, image2 = image1[None].cuda(), image2[None].cuda()
        image1, image2 = image1[None].cuda(), image2[None].cuda()

        flows = 0
        flow_count = 0

        this_flow_low = []
        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            flow_pre, flow_low = model(image1_tile, image2_tile)
            this_flow_low.append(forward_interpolate(flow_low[0])[None].cuda())

            padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].cpu()

        # _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5
        # print(f"valid_gt shape = {valid_gt.shape}, epe shape = {epe.shape}")

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        print(f"epe = {epe[val].mean().item()}")
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())


        # flow_img = flow_viz.flow_to_image(flow.permute(1, 2, 0).numpy())
        # image = Image.fromarray(flow_img)
        # if not os.path.exists(f'vis_kitti'):
        #     os.makedirs(f'vis_kitti/flow')
        #     os.makedirs(f'vis_kitti/image')

        # image.save('vis_kitti/flow/{:03d}-{:.3f}.png'.format(val_id, epe[val].mean().item()))
        # imageio.imwrite('vis_kitti/image/{:03d}_0_{:.3f}.png'.format(val_id, epe[val].mean().item()), image1[0].cpu().permute(1, 2, 0).numpy())
        # imageio.imwrite('vis_kitti/image/{:03d}_1_{:.3f}.png'.format(val_id, epe[val].mean().item()), image2[0].cpu().permute(1, 2, 0).numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    cfg = get_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    print(cfg.model)
    print(TRAIN_SIZE)
    print("Parameter Count: %d" % count_parameters(model))
    model.cuda()
    model.eval()

    for sigma in [0.05]:
        eval_kitti_training_288960(model.module, sigma=sigma)


    exit()

