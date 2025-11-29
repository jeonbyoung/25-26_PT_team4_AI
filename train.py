import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from get_rays import get_ray
from get_rgb import get_rgb
from get_samples import get_samples
from data_loader import load_full_data
from set_device import set_device
from volume_rendering import volume_rendering

def mse2pnsr(mse):
      return -10 * torch.log10(mse)

# 검증용 img 1장(test의 0번째 이미지로) 렌더링 함수
@torch.no_grad() # 학습 아니니, grad 하지 말라는 표시
def rendering_one_img_for_test(model, idx=0, device = None, target='lego'):
      width, height = 800, 800
      rays_d, rays_o, img_file_path = get_ray(category='test', target=target, i=idx)

      rays_o = torch.from_numpy(rays_o.reshape(-1,3).copy()).float().to(device)
      rays_d = torch.from_numpy(rays_d.reshape(-1,3).copy()).float().to(device)

      true_img = get_rgb(img_file_path)

      # chunking으로 메모리 터지는 거 방지
      chunk_size = 4096
      all_rgb = []

      for i in range(0, rays_o.shape[0], chunk_size):
            batch_o = rays_o[i : i+chunk_size]
            batch_d = rays_d[i : i+chunk_size]
            
            # test니까 그냥 64개 포인트로 지정
            pts, t_vals = get_samples(batch_d, batch_o, num_of_samples = 64, mode='test')

            pts_flat = pts.reshape(-1,3)
            dirs_expanded = batch_d[:,None,:].expand_as(pts)
            dirs_flat = dirs_expanded.reshape(-1,3)

            raw_rgb, raw_sigma = model(pts_flat, dirs_flat)

            rgb_for_vr = raw_rgb.reshape(batch_o.shape[0],64,3)
            sigma_for_vr = raw_sigma.reshape(batch_o.shape[0],64)

            rgb_chunk = volume_rendering(rgb_for_vr, sigma_for_vr, t_vals)

            all_rgb.append(rgb_chunk.cpu())

      pred_img= torch.cat(all_rgb, dim=0).reshape(height, width, 3).numpy()

      pred_img = np.clip(pred_img, 0 ,1)

      return pred_img, true_img



def train(model=None, optimizer=None, target = 'lego'):
      device = set_device()

      model = model.to(device)

      num_img = 100
      num_of_pts_per_ray = 64

      merged_ray_o, merged_ray_d, merged_rgb = load_full_data(num_img, target)


      # 이제 학습 시작
      # epoch도 설정하고, sample을 몇 개 쓸 지도 결정하면 된다.
      epoch = 100000
      num_of_rays = 2048
      start_epoch = 0

      # 노트북 발열이 심해서 잠깐 멈췄다. 아래는 그동안 학습한 거 저장한 걸 가지고 이어나가는 코드다.
      # 아래 resume_path는 직접 썼다. 나중에 중단 포인트가 바뀌면 변경해서 하면 됨.
      resume_path = 'NeRF_weights/NeRF_weights_n.pth'

      if os.path.exists(resume_path):
            print(f"Resuming training from {resume_path}")
            
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint)

            try:
                  loaded_epoch = int(resume_path.split('_')[-1].replace('.pth',''))
                  start_epoch = loaded_epoch + 1
                  print(f"Train restarted from {start_epoch}!")

            except:
                  print("Cannot read the epoch_num. Restart from 0 epoch.")
                  start_epoch = 0

      else:
            print('Train start from scratch! No check point found')

      pbar = tqdm(range(start_epoch,epoch),ncols=100)

      for i in pbar:
            idx = np.random.choice(len(merged_ray_d),num_of_rays)

            batch_o = merged_ray_o[idx].to(device)
            batch_d = merged_ray_d[idx].to(device)
            batch_rgb = merged_rgb[idx].to(device)

            pts, pts_dist_info = get_samples(batch_d, batch_o, mode='train')


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
                  pbar.set_postfix({'Epoch': i,'Loss': loss.item()})

            # 이전 gradient값 초기화
            optimizer.zero_grad()

            loss.backward()
            # 여기서 의문점이 들었다.
            # loss를 optimizer가 이렇게 되면 어찌 아는가?
            # 영특한 optimizer님은 알아서 안다고 하신다.
            # 정확히는 loss가 torch안에서 w.grad라는 변수에 저장된다고 한다.

            # optimizer가 이것을 읽어서 오류를 반영해준다.
            optimizer.step()

            # 진짜와 비교 and 가중치 저장
            if i%1000 == 0 and i >0:
                  # PSNR과 진짜 이미지와의 비교를 통해, 직접 얼마나 성장했나 보기
                  psnr_val = mse2pnsr(loss).item()

                  model.eval()
                  with torch.no_grad():
                        pred_img, true_img = rendering_one_img_for_test(model, idx=0, device = device, target=target)
                  model.train()

                  pbar.set_postfix({'Loss':f'{loss.item():.4f}', 'PSNR' : f'{psnr_val:.2f}'})

                  combined_img = np.hstack((pred_img, true_img))

                  save_dir = 'test_img'
                  if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                  plt.figure(figsize=(10,5))

                  plt.imshow(combined_img)
                  plt.text(10,700+60, f"Epoch: {i}\nPSNR: {psnr_val:.2f} dB",
                           color = 'yellow', fontsize=12, fontweight = 'bold',
                           bbox = dict(facecolor='black', alpha= 0.5))
                  
                  plt.text(10, 30, 'Prediction', color = 'black', fontweight = 'bold')
                  plt.text(800+10, 30, "Truth", color = 'black', fontweight ='bold')

                  plt.axis('off')


                  save_path = f"test_img/test_{i}_epoch.png"
                  plt.savefig(save_path, bbox_inches='tight', pad_inches= 0)
                  plt.close()


                  # 가중치 저장
                  save_dir = 'NeRF_weights'
        
                  if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                  file_path = os.path.join(save_dir, f"NeRF_weights_{i}.pth")

                  torch.save(model.state_dict(), file_path)



        

