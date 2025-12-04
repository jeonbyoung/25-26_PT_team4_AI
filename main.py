import torch

from set_device import set_device
from Model import Customed_NeRF
from train import train
from render_video import render_video

if __name__=="__main__":
    device = set_device()

    my_nerf = Customed_NeRF().to(device)
    my_optimizer = torch.optim.Adam(my_nerf.parameters(), lr=1e-3)

    train(my_nerf, my_optimizer, target='lego')

    render_video(model=my_nerf, save_path='first_test_result')