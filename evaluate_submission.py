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
from configs.sintel_submission import get_cfg as get_sintel_cfg
#from configs.kitti_submission import get_cfg as get_kitti_cfg
from core.utils.misc import process_cfg
import datasets
from utils import flow_viz
from utils import frame_utils
import imageio
# from FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer
from raft import RAFT

from utils.utils import InputPadder, forward_interpolate
import itertools

TRAIN_SIZE = [432, 960]

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    #ws.append(32)
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
def create_sintel_submission(model, output_path='output_path', sigma=1.0):
    """ Create submission for the Sintel leaderboard """
    print("no warm start")
    #print(f"output path: {output_path}")
    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    print(hws)

    results = {}

    model.eval()
    for dstype in ['final','clean']:
        test_dataset = datasets.MpiSintel_submission(split='test', aug_params=None, dstype=dstype, root="/mnt/lustre/shixiaoyu1/data/Sintel-test")
        epe_list = []
        for test_id in range(len(test_dataset)):
            if (test_id+1) % 100 == 0:
                print(f"{test_id} / {len(test_dataset)}")
                # break
            image1, image2, (sequence, frame) = test_dataset[test_id]
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
            flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

            #flow_img = flow_viz.flow_to_image(flow)
            #image = Image.fromarray(flow_img)
            #if not os.path.exists(f'vis_sintel'):
            #    os.makedirs(f'vis_sintel/flow')
            #    os.makedirs(f'vis_sintel/image')
                
            #image.save(f'vis_sintel/flow/{test_id}.png')
            #imageio.imwrite(f'vis_sintel/image/{test_id}_0.png', image1[0].cpu().permute(1, 2, 0).numpy())
            #imageio.imwrite(f'vis_sintel/image/{test_id}_1.png', image2[0].cpu().permute(1, 2, 0).numpy())

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
               os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    cfg = get_sintel_cfg()
    cfg.update(vars(args))
    print(cfg.model)
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model), strict=True)


    model.cuda()
    model.eval()

    #create_sintel_submission_notile(model.module, output_path="10_08_23_58-notile")
    #exit()
    
    for sigma in [0.05]:
        create_sintel_submission(model.module, output_path='output_path', sigma=sigma)



