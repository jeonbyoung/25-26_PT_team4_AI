import numpy as np
import torch
from tqdm import tqdm

from get_rays import get_ray
from get_rgb import get_rgb
from get_samples import get_samples
from data_loader import load_full_data
from set_device import set_device
from volume_rendering import volume_rendering

def train(model=None, optimizer=None):
        device = set_device()

        model = model.to(device)

        num_img = 100
        num_of_pts_per_ray = 64

        merged_ray_o, merged_ray_d, merged_rgb = load_full_data(num_img, target='lego')


        # 이제 학습 시작
        # epoch도 설정하고, sample을 몇 개 쓸 지도 결정하면 된다.
        epoch = 100000
        num_of_rays = 1024

        pbar = tqdm(range(epoch))

        for i in pbar:
            idx = np.random.choice(len(merged_ray_d),num_of_rays)

            batch_o = merged_ray_o[idx].to(device)
            batch_d = merged_ray_d[idx].to(device)
            batch_rgb = merged_rgb[idx].to(device)

            pts, pts_dist_info = get_samples(batch_d, batch_o, num_of_rays)


            # 여기부터 조금 어렵다...
            # Non-Lambertian Effect에 대한 내용을 담아서 pts를 변형해줘야한다.
            # Non-Lambertian Effect란 보는 각도에 따라서 해당 물체의 색이 변한다는 내용이다.
            # 지금 뽑아낸 pts는 그냥 공간 상의 '위치'를 나타낸 값이다. 즉, 이 점을 어디서 바라보고 있는 지에 대한 것도 넣어줘야한다는 것이다.
            # 그 값이 결국엔 model의 input이 되는, [위치 | 바라보는 각도인, view] 형태를 구현하는 것이다.
            
            # 그리고 batch_d는 그냥 본인의 방향만 나타내고 있는 1024 * 3의 형태다. 이걸 우선 pts와 모양을 맞춰줘야 한다.
            direction_expanded = batch_d[:,None,:].expand_as(pts)

            # 그럼 방향 행렬과 pts는 현재, 1024 * 64 * 3(x,y,z) 형태이다. 저 앞에 1024 * 64의 점들을 합쳐서 dim = 2로 만들어줘야한다.
            pts_for_model = pts.reshape(-1,3)
            dir_for_model = direction_expanded.reshape(-1,3)

            # 이제 (pts_for_model, dir_for_model)이라는 페어가 만들어졌고, 이 값이 model의 input으로서 들어가진다.
            from_model_rgb, from_model_sigma = model.forward(pts_for_model,dir_for_model)


            # 이렇게 Model을 거치고 온 pred_rgb, pred_sigma를 가지고, volume_rendering을 해야한다. 그 결과를 통해 나온 color를 가지고 실제 값과 비교할 것이니.
            # 그래서 우선 volume rendering을 위해, 얘가 어느 view에서 나온 건지 확인하기 위해, 다시 ray 단위로 묶는다.
            rgb_for_vr = from_model_rgb.reshape(num_of_rays, num_of_pts_per_ray, 3) # 이때의 3은 rgb 값
            sigma_for_vr = from_model_sigma.reshape(num_of_rays,num_of_pts_per_ray) # density에 해당하는 sigma는 스칼라다.

            # 이제 volume rendering을 할 차례다.
            pred_rgb = volume_rendering(rgb_for_vr,sigma_for_vr,pts_dist_info)

            # loss도 구하자. using MSE
            loss = torch.mean((pred_rgb - batch_rgb)**2)
            
            if(i%10==0):
                  pbar.set_description(f"Epoch {i}")
                  pbar.set_postfix({'Loss': loss.item()})

            # 이전 gradient값 초기화
            optimizer.zero_grad()

            loss.backward()
            # 여기서 의문점이 들었다.
            # loss를 optimizer가 이렇게 되면 어찌 아는가?
            # 영특한 optimizer님은 알아서 안다고 하신다.
            # 정확히는 loss가 torch안에서 w.grad라는 변수에 저장된다고 한다.

            # optimizer가 이것을 읽어서 오류를 반영해준다.
            optimizer.step()








        

