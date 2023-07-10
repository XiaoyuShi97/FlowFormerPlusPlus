import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
#from utils import flow_transforms 

from torchvision.utils import save_image

from utils import flow_viz
import cv2
from utils.utils import coords_grid, bilinear_sampler
import pickle

class FlowDataset(data.Dataset):
    def __init__(self):

        self.is_test = True
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        return img1, img2, self.extra_info[index]

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class TikTok_dataset(FlowDataset):
    def __init__(self):
        super(TikTok_dataset, self).__init__()

        root = 's3://'

        with open('./flow_dataset/TikTok_metafile.pkl', 'rb') as f:
            data = pickle.load(f)
        
        data = data[:1]

        for sequence in data:
            #print(len(sequence), sequence[0])
            self.image_list += [[root+sequence[0], root+sequence[4]]]
            self.extra_info += [(sequence[0], sequence[1])] # scene and frame_id
        
def fetch_dataloader():

    train_dataset = TikTok_dataset()

    train_loader = data.DataLoader(train_dataset, batch_size=1, pin_memory=False, shuffle=False, num_workers=8, drop_last=False)

    return train_loader
