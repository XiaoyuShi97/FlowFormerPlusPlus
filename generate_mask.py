import torch
import random
import numpy as np

H1, W1 = 46, 62
H2, W2 = 6, 8

grid_low = 4
grid_high = 15

for idx in range(100000):
    if idx % 1000 == 0:
        print(idx)
    gh = random.randint(grid_low, grid_high)
    gw = random.randint(grid_low, grid_high)

    down_sampled_noise = torch.rand(H1//gh+2, W1//gw+2, 1, H2*W2)
    down_sampled_noise = down_sampled_noise.repeat(1, 1, gh*gw, 1)
    up_sampled_noise = down_sampled_noise.reshape(H1//gh+2,  W1//gw+2, gh, gw, H2*W2).permute(0,2,1,3,4).reshape((H1//gh+2)*gh, (W1//gw+2)*gw, H2*W2)

    start_h = random.randint(0, (H1//gh+2)*gh-H1-1)
    start_w = random.randint(0, (W1//gw+2)*gw-W1-1)

    croped = up_sampled_noise[start_h:start_h+H1, start_w:start_w+W1, :]

    np.save('mae_mask/mask_46_62_48_{:06d}.npy'.format(idx), croped.numpy())


