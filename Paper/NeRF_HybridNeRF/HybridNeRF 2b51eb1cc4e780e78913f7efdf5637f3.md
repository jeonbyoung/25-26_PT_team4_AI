# HybridNeRF

https://arxiv.org/html/2312.03160v1

**목차**

# **HybridNeRF: Efficient Neural Rendering via Adaptive Volumetric Surfaces**

# Abstract

- 기존 NeRF는 최고 수준의 뷰 합성 품질을 제공하지만, volume rendering 방식(많은 샘플 및 쿼리 필요)으로 인해 렌더링 속도가 느림
- HybridNeRF: 대부분의 실제 물체는 볼륨 대신 표면으로 더 효율적으로 모델링할 수 있어 ray당 필요한 샘플 수가 훨씬 적음. 대부분을 표면 기반으로, 복잡한 부분만 볼륨 기반으로 렌더링

![HybridNeRF. 표면성 매개변수를 통해 하이브리드 표면-체적 표현을 훈련하여 적은 샘플로 장면 대부분을 렌더링.
하단 - 광선당 샘플 수를 시각화(밝을수록 많음).](HybridNeRF/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-24_15.25.36.png)

HybridNeRF. 표면성 매개변수를 통해 하이브리드 표면-체적 표현을 훈련하여 적은 샘플로 장면 대부분을 렌더링.
하단 - 광선당 샘플 수를 시각화(밝을수록 많음).

# Introduction

- **β - 표면성 매개변수**
    - 핵심 변환 매개변수. β-scaled SDF 값을 밀도로 변환하는 데 사용
    - β ↑ 하면 밀도가 표면 근처에 집중되어 **샘플 수↓**, 하지만 품질이 나빠짐.
    - HybridNeRF는 **x 위치마다 다른 β(x)** 를 사용함
    - **β(x)가 큰 곳**
        - SDF 형태가 정확함
        - 샘플 적게 사용 가능
    - **β(x)가 작은 곳**
        - 반투명, 얇은 구조, 유리 등
        - volume 특성을 유지해야 품질 유지
    
    👉 이 spatially adaptive β(x)가 HybridNeRF의 핵심.
    
1. **매개변수 β를 3D 장면 내 영역의 표면 특성에 대응하는 공간적으로 변화하는 매개변수 β(x)로 대체 → 장면의 대부분을 표면으로 효율적 모델링 가능**
2. **별도의 배경 모델 없이도 고품질의 복잡한 배경을 렌더링할 수 있도록 가중 Eikonal reqularization**
3. **hardware texture interpolation 및 구체 추적과 같은 특정 렌더링 최적화를 구현하여 고해상도 렌더링 속도를 크게 가속화**

### 접근법

![스크린샷 2025-11-24 15.54.17.png](HybridNeRF/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-24_15.54.17.png)

1. 배경 모델링 위해 VolSDF 유사 모델(거리 조정된 Eikonal loss 사용) 훈련
2. uniform 표면성 매개변수 β에서 위치 의존적 β(x) 값으로 전환 → 대부분의 장면을 얇은 표면(적은 샘플 필요)으로 모델링하면서도 미세하고 반투명한 구조 근처의 품질 저하를 방지
3. 장면의 95% 이상에서 유효한 SDF로 동작하므로, 렌더링 시점에 구체 추적(c)과 하위 수준 최적화(하드웨어 텍스처 보간)를 병행하여 각 샘플을 최대한 효율적으로 쿼리

# Method

3.1절에서 모델 아키텍처와 첫 번째 훈련 단계를 제시

3.2절에서는 품질 저하 없이 렌더링 속도를 높이기 위한 모델 미세 조정

3.3절에서는 무한한 장면 모델링 방법을 논의

3.4절에서 최종 렌더링 시간 최적화 방안을 제시

## 3. 주요 기여 및 최적화 단계

1. **하이브리드 표현**: 대부분의 장면($>95$% )을 표면으로 모델링하여 샘플 수를 줄이고, 순수 표면 방식보다 높은 충실도를 달성.
2. **가중 Eikonal 정규화**: 별도의 배경 모델 없이 복잡한 배경을 고품질로 렌더링할 수 있도록 힘.
3. **렌더링 최적화**: 하드웨어 텍스처 보간 및 구면 추적(Sphere Tracing)을 구현하여 고해상도 렌더링 속도를 크게 향상함.
4. **파이프라인**:
    - **1단계 (표현 학습)**: 거리 조정된 Eikonal 손실을 사용하여 $*\bar{\beta}(\mathbf{x})*$(전역 $*\beta*$)로 VolSDF와 유사하게 학습한다.
        
        ![스크린샷 2025-11-24 18.21.03.png](HybridNeRF/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-24_18.21.03.png)
        
    - **2단계 (미세 조정)**: $*\beta(\mathbf{x})*$를 공간적으로 적응시키고, 렌더링 속도 향상을 위해 최적화한다.
    - **3단계 (렌더링)**: $*\beta(\mathbf{x})*$가 유효한 SDF 영역에서는 **구면 추적**을 사용하고, 그 외 영역에서는 미리 정의된 단계 크기로 폴백(fallback)한다.

![**Figure 3. Surfaces.**](HybridNeRF/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-25_15.25.05.png)

**Figure 3. Surfaces.**

## 3.1 Representation

- **NeRF**
    - 장면을 연속적인 volumetirc radiance field로 표현하며, 이 필드는 장면의 기하학적 구조와 시점에 따른 변하는 색상을 모델의 가중치 내에 인코딩함
    1.  **렌더링 과정**
        - 각 픽셀을 렌더링할 때 해당 픽셀에서 카메라 방향으로 나가는 광선(ray)을 따라 여러 점 $x_i$를 샘플링함
        - 샘플링 각 점마다 MLP를 호출해 **밀도** $\sigma_i:=\sigma(x_i)$와 **색상** $*c_i:=c(x_i,d_r)*$을 얻고 볼륨 렌더링을 수행함. ($d_r$은 광선 방향)
    2. **밀도 → 불투명도 변화 (volume rendering)**
        - 밀도는 $\alpha_i:=1-exp(-\sigma_i\delta_i)$ 로 변환됨. ($\delta_i$는 샘플 간 거리)
    3. **최종 픽셀의 색상** -  ****$\hat{c_r}:=\sum\nolimits_{i=0}^{N-1} c_iw_i$
        - 가중치 $w_i := exp(-\sum\nolimits_{j=0}^{i-1} \sigma_j\delta_j)\alpha_i$
    - 훈련 - 이미지 픽셀을 배치로 샘플링한 뒤, 예측된 색상과 실제 색상 사이의 L2 재구성 손실을 최소화하도록 모델을 최적화
- **Modeling density -** SDF 기반 NeRF
    1. **표면-볼륨 하이브리드**: MLP 출력을 **부호화된 거리 함수(SDF)** $*f(x)*$로 해석하여 표면을 정의하고, 이를 밀도로 변환함.
        - SDF는 다음 조건을 만족
            - $f(x) = 0$ → 표면. 내부는 음수, 외부는 양수
            - 표면 주변에서는 $\|\nabla{f}\| = 1$ (signed distance property)
    2. **Eikonal 정규화**: SDF의 기울기 노름이 1이 되도록 $\mathcal{L}_{\text{Eik}}$손실을 사용하여 기하학적 품질을 개선한다.
        - MLP는 **Eikonal loss**로 정규화 (SDF는 gradient의 norm이 일반적으로 1)
            
            $$
            	\mathcal{L}_{\text{Eik}}(\mathbf{r}):=\sum_{i=0}^{N-1} \eta_i(\|\nabla{f(x_i)}\|-1)^2
            $$
            
            - $\eta_i$는 샘플마다 적용되는 가중치. 보통 1
    3. **표면성($\beta$)**: SDF 값 *f*(*x*)를 밀도 $\sigma$로 변환할 때 사용되는 파라미터
        - HybridNeRF는 이를 **공간적으로 가변적인 파라미터** $*\beta(\mathbf{x})*$로 대체한다.
        - SDF값을 밀도로
            
            $$
            ⁍
            $$
            
        - SDF
            
            $$
            \Psi(s)=\begin{cases}\frac{1}{2}exp(-s) & if\; s>0 \\
            1-\frac{1}{2}exp(s) & if\; s\le0.
            \end{cases}
            $$
            
- **Model Architecture**
    - HybridNeRF가 어떤 입력을 MLP에 넣어서 SDF와 색상을 예측하는가?
        - 3D feature grid와 triplane에서 multi-resolution feature를 가져와 합산(summing)하고, 이를 16차원으로 만들어 SDF/색상 MLP에 넣어 빠르고 정확한 3D 표현을 가능하게 함.
    - 3D 샘플 위치 임베딩
        1. **Dense 3D Feature Grid (3가지 해상도)**
            1. 3D 공간을 직접 voxel 형태로 저장. 정밀한(local) 공간 구조 표현에 강함
            2. coarse($128^3$) / mid($256^3$) / fine($512^3)$
        2. **Triplane**
            1. 3개의 2D 평면으로 3D 정보를 표현. 큰 구조(global feature) 표현에 강함
            2. XY / YZ / ZX plane
        - 한 점에서 **“3D grid + 3개의 2D triplane” 총 4개 feature → K=4**
    - 각 레벨에서 얻은 feature sum하며, 그 후 3D grid로부터 합산된 feature와 triplane에서 얻은 feature를 concat함. → **4K = 16 차원의 MLP 입력 벡터**를 만듦.
    - **viewing direction**은 **spherical harmonics degree 4**로 인코딩해 color MLP에 넣음
    - Aliasing 방지 - VR-NeRF 방식 사용
- **Optimization**
    - 최종 손실 함수
        
        $$
        ⁍
        $$
        
        - photometric loss ($L_{photo}$), Eikonal loss ($L_{Eik}$), interlevel loss ($L_{prop}$)

## 3.2. Finetuning - 적응형 표면성 $*\beta(\mathbf{x})*$ 학습

![**Figure 4. Choice of $\beta$**](HybridNeRF/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-25_15.26.43.png)

**Figure 4. Choice of $\beta$**

1. **$\beta$를 공간별로 다르게 조정**
    1. $*\beta(\mathbf{x})*$를 **$512^3$ voxel grid**로 구현. 각 voxel마다 $\beta v$ 값을 하나 가지고 있음
    2. fine-tuning 동안 모든 sample에 대해 (x, η, w)를 저장
        - $x_i$: 샘플 위치
        - $η$: Eikonal loss weight
        - $w$: ray marching weight(해당 샘플이 얼마나 중요한지)
2. **각 voxel에서 β를 증가시킬지 결정하는 조건**
    1. 해당 voxel의 Eikonal loss 평균(가중합 형태)
        
        $$
        avg=\frac{\sum w\eta(\|\nabla{f(x_i)}\|-1)^2}{\sum w}
        $$
        
        - 값이 0.25보다 작다면 → SDF 형상이 충분히 안정됨. → β를 증가시켜도 품질이 안 깨짐
        
        ![스크린샷 2025-11-25 15.29.05.png](HybridNeRF/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-25_15.29.05.png)
        
3. **βv 를 100씩 증가**
    1. SDF가 안정된($∥∇f∥≈1$) 영역 → β 높임 → 표면 기반 렌더링 가능 → 빠름
    2. SDF가 불안정한 영역(투명/복잡/얇은 구조) → β 증가 안 함 → volumetric으로 유지 → 품질 유지
4. **Proposal network baking**: NeRF에서 어디에서 샘플링할지를 정하기 위한 보조 네트워크
    1. Proposal network는 학습 때만 사용하고, 추론에서는 계산 비용을 줄이기 위해 이를 **1024³ occupancy grid로 미리 bake하고**, 이를 기반으로 샘플링해 **실시간 렌더링을 가능하게 함.**
5. **MLP 증류 (Distillation)**: 초기 학습에 사용된 큰 256 채널 MLP $*f*$를 더 작은 **16 채널 MLP $f_{\text{small}}$**로 증류하여 실시간 성능을 확보함.

## 3.3. Backgrounds

![스크린샷 2025-11-25 15.29.29.png](HybridNeRF/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-25_15.29.29.png)

- **문제**: 표면-볼륨 방식은 복잡한 배경 재구성에 어려움을 겪는다.
- **해결**: 단일 MLP가 전경에서는 SDF처럼, 배경에서는 볼륨 모델처럼 작동하도록 한다.
- **핵심 통찰**: $*\sigma(\mathbf{x})*$와 $\Psi(f(\mathbf{x}))$는 CDF $\Psi$ 덕분에 기능적으로 동등하며, Eikonal 정규화가 없으면 "SDF" MLP도 밀도 MLP처럼 작동할 수 있다.
1. **Relation between volumetric and surface-based NeRFs**: NeRF와 SDF NeRF의 차이는  **Eikonal 규제를 적용하느냐의 차이**
2. **Distance-adjusted loss**: 거리 $d_i$를 기반으로 Eikonal loss의 영향을 조절
    
    $$
    \eta_i=\frac{1}{d_i^2}
    $$
    
    - FG: $\eta$ 큼 → Eikonal 효과 강함 → SDF처럼 표면을 뚜렷하게 학습
    - BG: $\eta$ 작음 → Eikonal 효과 거의 없음 → NeRF처럼 자유롭게 volumetric 표현
    - 이는 전경에서는 유효한 SDF처럼, 배경에서는 NeRF처럼 행동하도록 유도하여 별도 모델 없이 고품질 배경 재구성을 가능하게 한다.

## 3.4. Real-Time Rendering

1. **텍스처 저장**: HybridNeRF는 multi-resolution feature를 미리 합산하여(3D grid와 Triplane 피처처) GPU texture로 저장함으로써, texture fetch 횟수를 3배 줄이고 하드웨어 interpolation을 활용해 실시간 렌더링 성능을 획기적으로 향상함.
    - 피처를 레벨별로 미리 합산하여 저장함으로써, MLP 평가당 텍스처 조회를 3배($24 \to 8$ 쿼리) 줄인다.
2. **구면 추적 (Sphere Tracing)**: SDF로 동작하는 영역($*\beta(\mathbf{x}_i) > 350*$일 때)에서 sphere tracing을 사용해 표면까지 빠르게 전진하고, 그렇지 않은 영역에서는 기존의 step size 기반 샘플링을 사용하여 속도와 품질을 모두 확보한다.

# 실험 결과 및 진단

1. **주요 성능 비교 (Eyeful Tower)**:
    - HybridNeRF는 VR 속도 목표(36FPS)를 달성한 몇 안 되는 방법 중 하나이며, 품질(PSNR)도 기존 최고치 대비 $>1.5\text{dB}$ 향상됨.
    - VR-NeRF(6 FPS), VolSDF(15 FPS) 대비 HybridNeRF는 **46 FPS**를 달성함.
2. **ScanNet++ 결과**: 3D Gaussian Splatting이나 MERF와 유사한 속도를 달성하면서도, 반사 표면이나 배경 재구성에서 가장 우수한 성능을 보임.
3. **진단 분석 (Ablation Study)**:
    - **적응형 표면성 $\beta(\mathbf{x})$**: 필수적이며, 전역 $\beta$사용 시 속도 또는 품질이 저하.
    - **거리 조정 Eikonal 손실**: 무한 장면에서 정확도를 보장하는 데 중요.
    - **MLP 증류 및 하드웨어 가속**: 품질 저하가 미미한 수준에서 렌더링 속도를 두 배로 증가시킴.

![스크린샷 2025-11-25 15.31.08.png](HybridNeRF/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-25_15.31.08.png)

![스크린샷 2025-11-25 15.31.22.png](HybridNeRF/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-25_15.31.22.png)

# 한계점 및 결론

1. **한계점**:
    - **메모리**: 3D 그리드 및 Triplane 사용으로 해시 테이블 방식보다 메모리 사용량이 많음.
    - **학습 시간**: Eikonal 정규화로 인해 iNGP보다 약 2배 느림.
2. **결론**: HybridNeRF는 표면 및 볼륨 렌더링의 장점을 결합하여 **최고 품질과 실시간 프레임 속도**를 동시에 달성함.
3. **향후 방향**: 스플래팅 기반 접근 방식과의 속도 격차를 줄이기 위해 두 방법론의 장점을 결합하는 것이 가치 있는 다음 단계.

# 부록 및 추가 정보

1. **색상 MLP 증류**: 색상 MLP를 64채널에서 16채널로 증류 시 46 FPS에서 60 FPS로 속도가 크게 향상되었으나 품질은 약간 저하되었다.
2. **Anti-Aliasing**: VR-NeRF와 유사하게 픽셀 풋프린트에 기반하여 그리드 피처를 감쇠시키는 전략을 사용한다.
3. **사회적 영향**: 고품질 신경 표현의 빠른 생성을 촉진하므로, 민감 정보(얼굴 특징, 번호판) 캡처와 관련된 **개인 정보 보호 및 보안 위험**이 존재한다.