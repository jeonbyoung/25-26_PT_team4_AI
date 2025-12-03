import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import imageio

from get_rays import get_rays
from render import render

def train(model, optimizer, images, poses, focal, H, W, testimg, testpose, device, n_iters=5001):
    # PSNR 계산 람다 함수
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(device)
    
    psnrs = []
    iternums = []

    plot_step = 500
    n_samples = 64

    plt.ion()
    fig = plt.figure(figsize=(12, 4))

    pbar = tqdm(range(n_iters), desc="Training NeRF")
    for i in pbar:
        img_i = np.random.randint(images.shape[0])
        target = images[img_i]
        pose = poses[img_i]

        pose = pose.to(device)

        # Core optimization loop
        rays_o, rays_d = get_rays(H, W, focal, pose)
        rgb, disp, acc = render(model, rays_o, rays_d, near=2., far=6., n_samples=n_samples, device=device, rand=True)
        
        optimizer.zero_grad()
        image_loss = torch.nn.functional.mse_loss(rgb, target)
        image_loss.backward()
        optimizer.step()

        psnr_curr = mse2psnr(image_loss).item()
        pbar.set_postfix({'PSNR': f'{psnr_curr:.2f}'})

        if i % plot_step == 0:
            with torch.no_grad():
                testpose = testpose.to(device)
                rays_o, rays_d = get_rays(H, W, focal, testpose)
                rgb, depth, acc = render(model, rays_o, rays_d, near=2., far=6., n_samples=n_samples, device=device)
                loss = torch.nn.functional.mse_loss(rgb, testimg)
                psnr = mse2psnr(loss).cpu().item()

                psnrs.append(psnr)
                iternums.append(i)
                
                plt.clf()
                
                # RGB 이미지
                plt.subplot(131)
                plt.imshow(rgb.cpu())
                plt.title(f"Iteration {i}")
                
                # PSNR 그래프
                plt.subplot(132)
                plt.plot(iternums, psnrs)
                plt.title("PSNR")
                plt.grid(True)
                
                # Depth Map
                plt.subplot(133)
                plt.imshow(depth.cpu(), cmap="gray")
                plt.title("Depth Map")

                plt.draw()
                plt.pause(0.01)

    print("✅ Training finished.")
    torch.save(model.state_dict(), "tiny_nerf_model.pt")
    
    print("🎥 Rendering 360 video...")
    frames = []
    
    render_poses = [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(0, 360, 120, endpoint=False)]
    
    for pose in tqdm(render_poses, desc="Video Frames"):
        pose = pose.to(device)
        with torch.no_grad():
            rays_o, rays_d = get_rays(H, W, focal, pose)
            rgb, _, _ = render(model, rays_o, rays_d, near=2., far=6., n_samples=n_samples, device=device)
            
            rgb = rgb.cpu().numpy()
            rgb8 = (255 * np.clip(rgb, 0, 1)).astype(np.uint8)
            frames.append(rgb8)

    imageio.mimwrite('result/video_result.mp4', frames, fps=30, quality=8)
    print("🎉 Video saved: video_result.mp4")
    
    plt.ioff()
    plt.show()

def trans_t(t):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

def rot_phi(phi):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

def rot_theta(th):
    return torch.tensor([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.tensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.float32) @ c2w
    return c2w