import torch

from set_device import set_device
from Model import Customed_NeRF
from train import train
from render_video import render_video

if __name__=="__main__":
    device = set_device()

    # 긴급 처방 후 나중에 보정 예정...
    """my_nerf = Customed_NeRF().to(device)
    my_optimizer = torch.optim.Adam(my_nerf.parameters(), lr=1e-3)

    train(my_nerf, my_optimizer, target='lego')

    render_video(model=my_nerf, save_path='first_test_result')"""

    model_coarse = Customed_NeRF(num_of_hidden_nodes=256).to(device)
    model_fine = Customed_NeRF(num_of_hidden_nodes=256).to(device)

    # [핵심 3] 두 모델의 파라미터를 모두 최적화 타겟으로 등록
    grad_vars = list(model_coarse.parameters()) + list(model_fine.parameters())
    
    # 학습률 5e-4 추천 (스케줄러와 함께 사용)
    my_optimizer = torch.optim.Adam(grad_vars, lr=5e-4)

    # train 함수 실행
    print("🚀 Coarse-to-Fine Training Start!")
    train(model_coarse, model_fine, my_optimizer, target='lego')