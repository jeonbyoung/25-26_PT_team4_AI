import torch
import numpy as np

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 장치 설정: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 장치 설정: Apple Silicon (MacBook M3)")
    else:
        device = torch.device("cpu")
        print("💻 장치 설정: CPU (GPU를 찾을 수 없음)")

    return device

def load_data(data_path, device):
    rawData = np.load(data_path)
    images = rawData["images"]
    focal = rawData["focal"]
    poses = rawData["poses"]

    img_H, img_W = images.shape[1:3]
    img_H = int(img_H)
    img_W = int(img_W)

    testimg, testpose = images[99], poses[99]

    images = torch.tensor(images, dtype=torch.float32, device=device)
    focal = torch.tensor(focal, dtype=torch.float32, device=device)
    poses = torch.tensor(poses,  dtype=torch.float32, device=device)
    testimg = torch.tensor(testimg, dtype=torch.float32, device=device)
    testpose = torch.tensor(testpose, dtype=torch.float32, device=device)

    return images, poses, focal, img_H, img_W, testimg, testpose