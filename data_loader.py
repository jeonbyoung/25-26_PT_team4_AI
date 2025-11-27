import torch
import numpy as np
from get_rays import get_ray
from get_rgb import get_rgb

def load_full_data(num_imgs=100, target='lego'):
    # 학습 데이터로 쓸 것들을 일단 다 뽑아낼 거다.
    # 그리고 각 지점에 대해 페어만 맞춰준 상태에서 얘네를 다 섞을 거다.
    # 생각해보면, 굳이 이미지 단위로 학습시킬 이유가 없다. 
    # 어짜피 한 시작점에서 어느 방향으로 가는 지, 그 지점의 색이 뭔지도 다 알고 있는 상황이니.

    every_ray_o = []
    every_ray_d = []
    every_rgb = []

    for i in range(num_imgs):
        ray_o, ray_d, img_path = get_ray('train', target,i)
            
        ray_o = ray_o.reshape(-1,3)
        ray_d = ray_d.reshape(-1,3)

        rgb = get_rgb(img_path).reshape(-1,3)

        every_ray_o.append(ray_o)
        every_ray_d.append(ray_d)
        every_rgb.append(rgb)

    merged_ray_o = torch.from_numpy(np.concatenate(every_ray_o, axis=0)).float()
    merged_ray_d = torch.from_numpy(np.concatenate(every_ray_d, axis=0)).float()
    merged_rgb = torch.from_numpy(np.concatenate(every_rgb, axis=0)).float()

    return merged_ray_o, merged_ray_d, merged_rgb