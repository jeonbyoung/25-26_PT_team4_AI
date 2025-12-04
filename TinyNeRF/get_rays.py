import torch
import numpy as np

def get_rays(img_H, img_W, focal, pose):
    device = pose.device

    i, j = torch.meshgrid(
        torch.arange(img_W, dtype=torch.float32, device=device),
        torch.arange(img_H, dtype=torch.float32, device=device), indexing='xy')
    
    # 카메라 좌표계에서의 방향 벡터 계산
    dirs = torch.stack(
      [(i-img_W*0.5)/focal,
       -(j-img_H*0.5)/focal,
       -torch.ones_like(i)], -1)
    dirs = dirs.to(device=device, dtype=pose.dtype)

    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)
    rays_o = pose[:3,-1].expand(rays_d.shape)
    
    return rays_o, rays_d