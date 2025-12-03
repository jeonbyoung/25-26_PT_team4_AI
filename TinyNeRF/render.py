import torch
import torch.nn.functional as F
from PE import positional_encoder

def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.

    return cumprod

def render(model, rays_o, rays_d, near, far, n_samples, device, rand=False):
    def batchify(fn, chunk=1024*32):
        return lambda inputs: torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    z = torch.linspace(near, far, n_samples).to(device)
    if rand:
        mids = 0.5 * (z[..., 1:] + z[...,:-1])
        upper = torch.cat([mids, z[...,-1:]], -1)
        lower = torch.cat([z[...,:1], mids], -1)
        t_rand = torch.rand(z.shape).to(device)
        z = lower + (upper - lower) * t_rand

    points = rays_o[..., None, :] + rays_d[..., None, :] * z[..., :, None]

    flat_points = torch.reshape(points, [-1, points.shape[-1]])
    flat_points = positional_encoder(flat_points)
    
    raw = batchify(model)(flat_points)
    raw = torch.reshape(raw, list(points.shape[:-1]) + [4])

    # Compute opacities and color
    sigma = F.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])

    # Volume Rendering
    one_e_10 = torch.tensor([1e10], dtype=rays_o.dtype).to(device)
    dists = torch.cat((z[..., 1:] - z[..., :-1],
                      one_e_10.expand(z[..., :1].shape)), dim=-1)
    alpha = 1. - torch.exp(-sigma * dists)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * z).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    
    return rgb_map, depth_map, acc_map