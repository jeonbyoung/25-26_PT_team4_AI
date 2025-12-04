import torch
import torch.nn as nn
import sys

from load_data import load_data, set_device
from model import TinyNerfModel
from train import train

def main():
    L_embed = 6
    lr = 5e-3
    n_iters = 30001

    # 1. Device 설정
    device = set_device()
    print(f"Using device: {device}")

    # 2. 데이터 로드 (경로는 실제 파일 위치로 수정 필요)
    data_path = "tiny_nerf_data.npz" 
    images, poses, focal, H, W, testimg, testpose = load_data(data_path, device)
    print(f"Data Loaded: Images {images.shape}, Focal {focal}")

    # 3. 모델 초기화
    nerf = TinyNerfModel(num_encoding_functions=L_embed)
    nerf = nerf.to(device)
    
    # 4. 옵티마이저 설정
    optimizer = torch.optim.Adam(nerf.parameters(), lr=lr, eps=1e-7)

    # 5. 학습 시작
    train(nerf, optimizer, images, poses, focal, H, W, testimg, testpose, device, n_iters=n_iters, L_embed=L_embed)

if __name__ == "__main__":
    main()