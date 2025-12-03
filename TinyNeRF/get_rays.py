import torch
import numpy as np

def get_rays(img_H, img_W, focal, c2w):
    device = c2w.device

    i, j = torch.meshgrid(
        torch.arange(img_W, dtype=torch.float32, device=device),
        torch.arange(img_H, dtype=torch.float32, device=device), indexing='xy')
    
    dirs = torch.stack(
      [(i-img_W*0.5)/focal,
       -(j-img_H*0.5)/focal,
       -torch.ones_like(i)], -1)
    
    dirs = dirs.to(device=device, dtype=c2w.dtype)

    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    
    return rays_o, rays_d