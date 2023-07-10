from __future__ import print_function, division
import sys
# sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core import optimizer
import core.pretrain_maemask_datasets as datasets
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger

# from torch.utils.tensorboard import SummaryWriter
from core.utils.logger import Logger

# from core.FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(cfg):
    model = nn.DataParallel(build_flowformer(cfg))
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, mask = [x.cuda() for x in data_blob]

            if cfg.add_noise:
                #print("[Adding noise]")
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            output = {}
            loss = model(image1, image2, mask=mask, output=output)
            loss = loss.mean()
            metrics = {"loss": loss.item()}
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            logger.push(metrics)

            total_steps += 1
            
            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='flowformer', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    args = parser.parse_args()

    if args.stage == 'youtube':
        from configs.pretrain_config import get_cfg
    

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    train(cfg)
