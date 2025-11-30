import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from get_rays import get_ray
from get_rgb import get_rgb
from get_samples import get_samples, sample_pdf # sample_pdf 추가
from data_loader import load_full_data
from set_device import set_device
from volume_rendering import volume_rendering

def mse2pnsr(mse):
      return -10 * torch.log10(mse)

# [수정] 모델 2개를 받도록 변경
@torch.no_grad()
def rendering_one_img_for_test(model_coarse, model_fine, idx=0, device=None, target='lego'):
      width, height = 800, 800
      rays_d, rays_o, img_file_path = get_ray(category='test', target=target, i=idx)

      rays_o = torch.from_numpy(rays_o.reshape(-1,3).copy()).float().to(device)
      rays_d = torch.from_numpy(rays_d.reshape(-1,3).copy()).float().to(device)

      true_img = get_rgb(img_file_path)

      # [수정] 메모리 절약을 위해 512로 줄임
      chunk_size = 512
      all_rgb = []

      for i in range(0, rays_o.shape[0], chunk_size):
            batch_o = rays_o[i : i+chunk_size]
            batch_d = rays_d[i : i+chunk_size]
            
            # ----------------------------------------
            # 1. Coarse 단계 (정찰)
            # ----------------------------------------
            pts_c, z_vals_c = get_samples(batch_d, batch_o, num_of_samples=64, mode='test')

            # Reshape for model
            pts_flat_c = pts_c.reshape(-1,3)
            dirs_expanded_c = batch_d[:,None,:].expand_as(pts_c)
            dirs_flat_c = dirs_expanded_c.reshape(-1,3)

            # Coarse 모델 실행
            raw_rgb_c, raw_sigma_c = model_coarse(pts_flat_c, dirs_flat_c)

            # Reshape back
            rgb_for_vr_c = raw_rgb_c.reshape(batch_o.shape[0], 64, 3)
            sigma_for_vr_c = raw_sigma_c.reshape(batch_o.shape[0], 64)

            # Rendering -> weights 추출
            _, weights_c = volume_rendering(rgb_for_vr_c, sigma_for_vr_c, z_vals_c)

            # ----------------------------------------
            # 2. Fine 단계 (저격)
            # ----------------------------------------
            # Coarse에서 얻은 weights를 기반으로 샘플 추가
            z_vals_mid = .5 * (z_vals_c[...,1:] + z_vals_c[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights_c[...,1:-1], 128, det=True)
            z_vals_fine, _ = torch.sort(torch.cat([z_vals_c, z_samples], -1), -1)

            # 좌표 다시 계산 (총 192개)
            pts_f = batch_o[...,None,:] + batch_d[...,None,:] * z_vals_fine[...,:,None]
            
            # Reshape for model
            pts_flat_f = pts_f.reshape(-1,3)
            dirs_expanded_f = batch_d[:,None,:].expand_as(pts_f)
            dirs_flat_f = dirs_expanded_f.reshape(-1,3)

            # Fine 모델 실행
            raw_rgb_f, raw_sigma_f = model_fine(pts_flat_f, dirs_flat_f)

            # Reshape back
            rgb_for_vr_f = raw_rgb_f.reshape(batch_o.shape[0], 64+128, 3)
            sigma_for_vr_f = raw_sigma_f.reshape(batch_o.shape[0], 64+128)

            # 최종 Rendering
            rgb_chunk, _ = volume_rendering(rgb_for_vr_f, sigma_for_vr_f, z_vals_fine)

            all_rgb.append(rgb_chunk.cpu())

      pred_img= torch.cat(all_rgb, dim=0).reshape(height, width, 3).numpy()
      pred_img = np.clip(pred_img, 0 ,1)

      return pred_img, true_img


# [수정] 인자 변경: model 1개 -> model_coarse, model_fine
def train(model_coarse=None, model_fine=None, optimizer=None, target = 'lego'):
      device = set_device()
      
      # 모델 GPU 이동은 main에서 했지만 혹시 모르니 확인
      # model_coarse = model_coarse.to(device)
      # model_fine = model_fine.to(device)

      # 스케줄러 설정 (optimizer에 두 모델 파라미터가 다 들어있어야 함 -> main.py에서 처리됨)
      total_steps = 200000
      scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1**(1/total_steps))

      num_img = 100
      
      # [설정] 샘플 개수 분리
      N_coarse = 64
      N_fine = 128

      merged_ray_o, merged_ray_d, merged_rgb = load_full_data(num_img, target)

      epoch = 200000
      num_of_rays = 512 # 메모리 보호를 위해 512 유지
      start_epoch = 0
      
      resume_path = 'NeRF_weights/NeRF_weights_n.pth'

      # [수정] Resume 로직 (두 모델 로드)
      if os.path.exists(resume_path):
            print(f"Resuming training from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # 저장할 때 키 값을 구분해서 저장하도록 수정할 예정임
            try:
                model_coarse.load_state_dict(checkpoint['coarse'])
                model_fine.load_state_dict(checkpoint['fine'])
                
                # 파일명에서 에폭 읽기 시도 (형식이 안 맞으면 0부터)
                # loaded_epoch = ... (복잡하니 생략, Scratch 권장)
                print("Models loaded successfully.")
            except:
                print("Checkpoint structure mismatch. Starting from scratch.")
                start_epoch = 0
      else:
            print('Train start from scratch! No check point found')

      pbar = tqdm(range(start_epoch, epoch), ncols=100)

      for i in pbar:
            idx = np.random.choice(len(merged_ray_d), num_of_rays)

            batch_o = merged_ray_o[idx].to(device)
            batch_d = merged_ray_d[idx].to(device)
            batch_rgb = merged_rgb[idx].to(device)

            # =================================================================
            # [Step 1] Coarse Model Execution (정찰)
            # =================================================================
            # 1. 샘플링 (64개) -> pts_dist_info_c는 z_vals를 의미함
            pts_c, pts_dist_info_c = get_samples(batch_d, batch_o, mode='train', num_of_samples=N_coarse)

            # 2. Reshape (모델 입력용)
            direction_expanded_c = batch_d[:,None,:].expand_as(pts_c)
            pts_for_model_c = pts_c.reshape(-1,3)
            dir_for_model_c = direction_expanded_c.reshape(-1,3)

            # 3. Model Forward
            from_model_rgb_c, from_model_sigma_c = model_coarse.forward(pts_for_model_c, dir_for_model_c)

            # 4. Reshape (렌더링용)
            rgb_for_vr_c = from_model_rgb_c.reshape(num_of_rays, N_coarse, 3)
            sigma_for_vr_c = from_model_sigma_c.reshape(num_of_rays, N_coarse)

            # 5. Volume Rendering -> weights_c 획득 (중요!)
            pred_rgb_c, weights_c = volume_rendering(rgb_for_vr_c, sigma_for_vr_c, pts_dist_info_c)

            # =================================================================
            # [Step 2] Fine Model Execution (저격)
            # =================================================================
            # 1. Coarse 결과(weights)를 바탕으로 샘플 추가 (128개)
            z_vals_mid = .5 * (pts_dist_info_c[...,1:] + pts_dist_info_c[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights_c[...,1:-1], N_fine, det=False)
            z_samples = z_samples.detach() # 미분 끊기

            # 2. 기존 샘플 + 새 샘플 합치기 (64 + 128 = 192개)
            pts_dist_info_f, _ = torch.sort(torch.cat([pts_dist_info_c, z_samples], -1), -1)

            # 3. 좌표(pts) 다시 계산
            pts_f = batch_o[...,None,:] + batch_d[...,None,:] * pts_dist_info_f[...,:,None]

            # 4. Reshape (모델 입력용)
            direction_expanded_f = batch_d[:,None,:].expand_as(pts_f)
            pts_for_model_f = pts_f.reshape(-1,3)
            dir_for_model_f = direction_expanded_f.reshape(-1,3)

            # 5. Model Forward (Fine 모델 사용)
            from_model_rgb_f, from_model_sigma_f = model_fine.forward(pts_for_model_f, dir_for_model_f)

            # 6. Reshape (렌더링용)
            rgb_for_vr_f = from_model_rgb_f.reshape(num_of_rays, N_coarse + N_fine, 3)
            sigma_for_vr_f = from_model_sigma_f.reshape(num_of_rays, N_coarse + N_fine)

            # 7. Volume Rendering -> 최종 RGB
            pred_rgb_f, _ = volume_rendering(rgb_for_vr_f, sigma_for_vr_f, pts_dist_info_f)

            # =================================================================
            # [Step 3] Loss Calculation & Optimization
            # =================================================================
            loss_c = torch.mean((pred_rgb_c - batch_rgb)**2)
            loss_f = torch.mean((pred_rgb_f - batch_rgb)**2)
            
            loss = loss_c + loss_f # 두 모델 다 학습

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging (Fine Loss 기준)
            if(i%10==0):
                  pbar.set_postfix({'Epoch': i,'Loss': loss_f.item()}) # Fine loss만 보는 게 정신건강에 좋음

            # OOM 방지
            del pred_rgb_c, pred_rgb_f, batch_rgb, pts_c, pts_f
            if i%100 == 0:
                  torch.cuda.empty_cache()

            # 검증 및 저장 (Fine Model 결과 사용)
            if i%1000 == 0 and i >0: # 1000번으로 주기 늘림
                  psnr_val = mse2pnsr(loss_f).item()

                  model_coarse.eval()
                  model_fine.eval()
                  
                  with torch.no_grad():
                        pred_img, true_img = rendering_one_img_for_test(model_coarse, model_fine, idx=0, device = device, target=target)
                  
                  model_coarse.train()
                  model_fine.train()

                  current_lr = optimizer.param_groups[0]['lr']
                  pbar.set_postfix({'Loss':f'{loss_f.item():.4f}', 
                                    'PSNR' : f'{psnr_val:.2f}',
                                    'LR': f'{current_lr:.6f}'
                              })

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

                  # 가중치 저장 (두 모델 다 저장)
                  save_dir = 'NeRF_weights'
                  if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                  file_path = os.path.join(save_dir, f"NeRF_weights_{i}.pth")
                  
                  # 딕셔너리로 묶어서 저장
                  torch.save({
                      'coarse': model_coarse.state_dict(),
                      'fine': model_fine.state_dict()
                  }, file_path)