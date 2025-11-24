# NeRF : Representing Scenes as Neural Radiance Fields for View Synthesis

https://www.matthewtancik.com/nerf

paper : https://arxiv.org/pdf/2003.08934

# #1 Intro

기존에는 Pixel, Voxel(CNN 처리에 용이), Point Cloud, Mesh 방식을 많이 사용!

![pixel, point cloud, mesh](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image.png)

pixel, point cloud, mesh

하지만, 

1. Voxel(3차원 공간을 구성하는 단위 정육면체)의 경우, cubic 크기만큼의 메모리 공간을 사용해서 부담이 컸음
2. point cloud, mesh의 경우, 사용하는 vertex의 개수를 한정지을 수 밖에 없는 단점이 있음

⇒ 최근 주목 받는 기술은 MLP기반, 연속함수를 activation func으로 사용하는 기술 : Implicit Representation(Coordinate-based Representation)

![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%201.png)

Simply thinking ⇒ 2D image의 x,y 좌표를 주고, 이걸 rgb 로 output을 받아내는 것.

⇒ 즉, 네트워크에 안에 이런 변환 과정을 기억시키는 것.

💡 만약 3D라면?

![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%202.png)

⇒ 3차원 공간 내에서 classification을 하게 됨

위 그림에서 왼쪽을 보면, 토끼의 안쪽에 있는 값은 $SDF>0$으로, 아닌 값들은 <0으로 잡아서, classification을 진행하게 됨.

💡오른쪽 마네킹같은 그림은, 1,4열에 해당하는 그림들이 input이고, 이를 토대로, 3D 복원 + color embedding까지 마친 것.

⇒ 맨 오른쪽 아래의 모델의 경우, 색등이 흐릿하게 표현되는 등, 애매함.

⇒ 이걸 어떻게 더 발전시킬 수 있을까?

![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%203.png)

💡 과연 input에는 공간적인 위치만 들어오고, output에는 RGB만 나오는 것이 맞는가?

![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%204.png)

⇒ input에는 viewing direction(maybe x-y, y-z)을 넣고, output에는 occupancy에 해당하는 $\sigma$를 넣어서, 값을 더 풍부하게 해줌.

즉, 3D 복원을 실행할 때, 2D에서 한 것처럼, 

2 dimension → (R,G,B)가 아닌, 

**3 dimension + 2 dim(viewing direction) → (R,G,B) + occupancy를 진행한 것!** 

그리고, 이런 모델이 학습되는 곳을 **Neural Radience Field**라고 부름!

⇒ 이것의 이점 : 사전 학습된 모델(Neural Radience Field)가 있다면, 별도로 여러 데이터들이 없더라도, viewing direction 등을 정해주기만 하면, 복원이 가능해짐!

# #2 NeRF

![NeRF로 3D 복원된 영상이였음. 실제론 막 돌아가는데, 마이크의 그리드, 레고의 볼록 튀어나온 부분, 음영등을 잘 복원해준다.](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%205.png)

NeRF로 3D 복원된 영상이였음. 실제론 막 돌아가는데, 마이크의 그리드, 레고의 볼록 튀어나온 부분, 음영등을 잘 복원해준다.

⇒ 완전 새 기술이 적용되기도 하지만, classic한 기술들도 적용됨.

## #2.1 Classic Volume Rendering

- Volume Ray Casting
    
    volume 정보를 가지고 있는 object로부터, 2D image를 뽑아내는 것.
    
    ⇒ 특정 view에서 3D object를 2D image로 렌더링했을 때, image는 어떤 color를 가질 것인가?를 특정하는 기술
    
    ![왼쪽 아래의 구는 우리가 바라보는 view](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%206.png)
    
    왼쪽 아래의 구는 우리가 바라보는 view
    
    $$
    C(r) = \int_{t_n}^{t_f}T(t)\sigma(r(t))c(r(t),d)~dt, ~where~~T(t)=exp(-\int_{t_n}^t\sigma(r(s))~ds)
    $$
    
    $C(r)$ : 2D image에 표시되는 색깔
    
    ![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%207.png)
    
    다음과 같이, view에서 뻗어나가는 방향벡터의 경우,
    
    - $o$ : 시작점(view point)
    - $d$  : 방향 벡터
    - $t_n$ : 3D object가 r 기준 view point에서 가장 가까운 지점
    - $t_n$ : 3D object가 r 기준 view point에서 가장 먼 지점
    - $t$ : 3D object의 속 임의 지점
    - $c(r(t),d)$ : 그 방향 벡터의 3D 지점에서의 color
    - $\sigma(r(t))$ : 그 지점에서의 occupancy를 나타내는 것. (⇒ 불투명도 높을 수록, 1에 가까워짐)
    - $T(t)$  : transmittance(tn ~ t까지 있는 것들)를 accumulate 해놓은 것.
        
        ⇒ 지점 $t$까지 도달하기에, 얼마나 불투명한 물체들이 가로막고 있는가를 기술하는 척도
        
        ![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%208.png)
        
    
    ⇒ 그럼 3D object의 모든 t를 찾아낼 수 있을 것인가?
    
    ⇒  그건 impossible ⇒ sampling을 통해 t들을 찾아냄.
    
    ![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%209.png)
    
    위에서 모든 3D points에 대해서는 그런 방식을 사용하고, samlping했을 때는 모두 반영된 것이 아니니, approximation을 하여, 식을 작성
    
    $$
    \hat{C}(r) = \Sigma_{i=1}^{N}T_i(1-exp(-\sigma_i\delta_i))c_i,~~where~~T_i = exp(-\Sigma_{j=1}^{i-1}\sigma_j\delta_j)
    $$
    
    ⇒ approx에서 달라지는 점은 occupancy에 대한 부분!
    
    - $\delta_i$ : 2번에서 나타나는 sampling돼서 나타난 점들 사이의 간격!
    
    ![y=x와 y=1-exp(-x)의 비교 ⇒ 값의 차이는 있으나, 경향성은 비슷함을 알 수 있다.](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%2010.png)
    
    y=x와 y=1-exp(-x)의 비교 ⇒ 값의 차이는 있으나, 경향성은 비슷함을 알 수 있다.
    
    - **Contribution Weight**
        - $C(r)$에서 $T(t)\sigma(r(t))$
        - $\hat{C}(r)$에서 $T_i(1-exp(-\sigma_i\delta_i))$
        
        **즉, $c$앞에 있는 값들, color가 얼마나 반영될 수 있는가를 나타내주는 값들.**
        

## #2.2 NeRF Training

![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%2011.png)

1. Viewpoint Selection
2. Ray Composition
3. Select 5D input samples along the ray (Coarse to Fine manner)
4. Query into MLP
5. Get predicted Color + Density
6. Render color using volume ray casting
7. Compute rendering loss(Simply squared error between rendered and true pixel colors)

⇒ 근데, 결국 우리는 sampling을 하게 되는데, 어떤 방식으로 할까?

⇒ Random, Uniform dist로 할 수도 있음.

⇒ 근데 필요한 건 **sampling시, high img color/geometry resolution을 가지는 중요한 포인트들(object boundary)**을 가져야할 필요가 있음! ⇒ Hierarchical volume sampling

- **Hierarchical Volume Sampling**
    
    ⇒ 2 step(**Coarse to Fine manner**)으로 ray를 따라 점들을 sampling함.
    
    1. **Coarse**
        
        ray가 지나가는 곳을 $N_c(c:coarse)$개의 section으로 나누고, 
        각 section마다 점을 하나씩 가져와서 sampling함. ⇒ 이런 sampling 방식을 **Stratified sampling**방식이라고 함.
        
        $$
        \hat{C}(r) = \Sigma_{i=1}^{N}T_i(1-exp(-\sigma_i\delta_i))c_i
        \\
        \downarrow normalization \downarrow
        \\
        \hat{w}_i = w_i/\Sigma_{j=1}^{N_c}w_j
        $$
        
        이렇게 뽑아낸 점들에 대해, MLP(FCN/Full connected Network)에 넣어서, contribution weight들을 가져오게 됨. 이때의 MLP를 Coarse Network라고 부름.
        
        이런 방식을 통해, contribution weight는 probability distribution을 따르게 됨!
        
    2. **Fine**
        
        **위에서 나온 probability distribution에 맞춰,** $N_f(f:fine)$개의 점들을 sampling함.
        
        ⇒ 이런 방식을 **Inverse transform sampling**이라고 함.
        
        - **🦖 More details about Inverse transform sampling**
            
            ⇒ particle search같은 분야에서 많이 쓰이는 최적화 sampling 기법이라고 함.
            
            ![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%2012.png)
            
            $y=F(x)$가 uniform distribution이고, F가 CDF of continuous X일 때, 
            
            + X가 Random variable이고, continuous하며, strictly increasing하는 CDF를 가질 때,
            
            CDF는 [0,1]의 uniform distribution을 가짐.
            
            ⇒ 즉, CDF를 봤을 때, 거의 시작하자마자 1로 올라감. why? ⇒ 밀도가 높은 건 다 앞에 있었다는 뜻! 앞에서 다 이미 채워졌다는 뜻.
            
            ⇒ 이런 부분을 가져오게 됨.
            
        
        ![왼쪽 두 그림이 coarse, 오른쪽 한 개의 그림이 Fine에 해당함.](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%2013.png)
        
        왼쪽 두 그림이 coarse, 오른쪽 한 개의 그림이 Fine에 해당함.
        
        ⇒ 위에서 나온 pdf(probability density func)를 바탕으로 cdf(cumulative distribution func) 를 구함. 이 cdf의 inverse를 구하여, sampling을 함.
        
         $pdf ->\int->cdf$
        
        cdf의 inverse의 정의역은 그에 대한 확률일 테고, 함수값은 그 안에 속한 점들의 수일테니, 해당하는 점들의 개수를 찾아낼 수 있음. 
        
        ![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%2014.png)
        
        그렇게 점들을 구해서, 총 $N_c+N_f$만큼의 점들을 가지고, MLP를 돌리게 됨. 이때의 **MLP를 fine network**라고 부름.
        
        그리고 MLP의 output으로, **color와 점들의 volume density**를 얻어내는 것!
        
    
1. Query into MLP
2. Get predicted Color + Density
    
    ![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%2015.png)
    
    총 $N_c+N_f$개의 input을 fine network에 집어넣음.
    
    - activation func : ReLU
    - 9개의 hidden layer를 사용하는데, 신기한 점은 앞의 8개의 layer를 통과하는 과정에서는 x,y,z 3가지 정보만 가지고 수행됨.
    - 그걸 마치고 난 뒤에 volume density에 해당하는 $\sigma$를 뽑아내고, 그것과 가지고 있던 viewing direction 값을 가지고 마지막 은닉층을 통과한 뒤에, RGB를 가지고 오게 됨.
    
    **⇒ But, Why? 처음부터 안 넣고, 나중에 넣고 하는 과정을 거치는가?**
    
    ⇒ Non-Lambertian effect
    
- **Non-Lambertian effect**
    
    cf) Lambertian effect(reflect) : 관찰자가 바라보는 각도와 관계없이 같은 겉보기 밝기를 갖는 것을 의미.
    
    ⇒ Non이 붙었으니, 그것이 아닌 것을 의미!
    
    실제로 어느정도 반사율을 가지는 물체의 경우, 보는 각도에 따라 컬러 등이 달라질 수 있음!
    
    ⇒ 근데 volume density는 보는 각도에 관계 없이 항상 동일해야됨!
    
    ![스크린샷 2025-11-11 오후 11.19.24.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.19.24.png)
    
    그래서 다음과 같은 방법을 가져감
    
    1. 8개의 FC layer를 가지고, volume density 값을 location값인 x,y,z만 가지고 뽑아냄.
    2. 1개의 FC layer로, color를 location값들이랑 viewing direction d를 가지고 뽑아냄.
    
1. Render color using volume ray casting
    
    1~5번 과정을 거치며, $N_c+N_f$개의 ray위에 있는 points를 얻어냄.
    
    approximation을 한 $\hat{C}(r)$값을 이용하여, color 값을 구하고,
    
    이는 대략, $C \approx \Sigma_{i=1}^{N}T_i\alpha_ic_i~~where ~\alpha_i=1-\exp(-\sigma_i\delta_i)$
    
    이때의 $T_i$는 weight, $c_i$ 는 color를 의미.
    
    $\alpha_i$는 빛을 받은 정도를 의미
    
    ⇒ 큰 계산없이, naturally differentiable하기 때문에, back propagation에 필요한 gradient도 잘 구해질 것임.
    

1. Compute rendering loss
    
    ⇒ 방식은 그냥 예측했던 컬러값 - 실제 컬러값 을 비교!
    
    해당 논문의 실험에서는 ray를 총 4096개 사용함.
    
    즉, view point를 4096개 사용한 것.
    
    각 view point당 뽑은 샘플의 개수는 256개라고 함.
    
    즉, $N_c+N_f$ 값이 256!
    
    결국, 총 실험간에 사용한 sample의 수가 4096*256
    
    즉, $2^{20}$개의 sample을 쓴 것.
    
    $\mathcal{L}=\Sigma_{r\in\mathcal{R}}[||\hat{C}_c(r)-C(r)||_2^2+||\hat{C}_f(r)-C(r)||_2^2]$
    
    위와 같은 loss function을 사용하는데, 
    
    - 앞부분은 Coarse network에 대한 Loss Rendering!
    - 뒷부분은 Fine network에 대한 Loss Rendering!
    
    - $\mathcal{R}$ : 각 batch당 가지는 ray(view point)의 수
    - $C(r)$ : r(ray)에 대한 진짜 RGB값
    - $\hat{C}_c(r)$ : r에 대해 coarse network가 추측한 color
    - $\hat{C}_f(r)$ : r에 대해 fine network가 추측한 color
    
    근데 coarse network에 대한 loss는 왜 쓸까?
    
    ⇒ 어짜피 coarse manner에서 구한 것을 토대로 inverse transform sampling한 게 fine network에서 구한 것인데?
    
    **⇒ 후에 coarse network에서 판단한 것을 기반으로 나중에 한 번 더 sample을 해서 쓴다고 함!**
    
    **⇒ 더 좋은 density&color를 estimate하도록 해야됨!**
    

## #2.3 NeRF Result

그렇게 1~7번을 마치고 나면, 왼쪽, Naive한 방식의 것을 얻어낼 수 있음

![스크린샷 2025-11-12 오전 12.03.57.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-12_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.03.57.png)

⇒ 자세히 보면, high frequency에 해당하는 얘들에 대한 구현이 잘 되지 않음!

⇒ 잎의 표현, 위에 에어컨의 디테일 등.

⇒ 논문의 저자들은 5D input을 그냥 그대로 MLP에 넣어버린 것이 이런 문제의 원인이라고 생각했음

⇒ high-frequency variation을 표현하기에는, deep한 NN이 lower frequency function에 bias돼있다고 생각했기 때문!

- **NeRF 이전에 등장했던 딥러닝 기반, multi-view image를 이용한 NVS(Novel View Synthesis)에서 사용되는 것들(SRN,LLFF,NV)**
    
    ![스크린샷 2025-11-12 오전 12.16.42.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-12_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.16.42.png)
    
    1. SRN(Scene Representation Networks)
        
        ⇒ 불투명한 표면의 데이터를 표현하는 데에 중점을 둔 모델.
        
        특정 3D 좌표의 feature vector를 예측하고, 이것으로부터 해당 좌표의 color를 예측하는 방식으로 작동
        
        - 단점
            - 한 개의 scene을 학습하는 데에 최소 12시간 이상이 소요됨.
        
    2. LLFF(Local Light Field Fusion)
        
        ⇒ 여러 개의 input img를 기반으로 현실적인 새로운 시점을 생성하기 위해 설계된 모델.
        
        3D voxel grid 또는 다시점 입체 이미지(MPI, Multiplane Images)를 활용하여, 장면을 표현하고, 이것들을 합성하여, 새로운 뷰를 렌더링하는 기술.
        
        - 단점
            
            SRN에 비해 학습 시간이 짧으나,(10분 이내) 모든 입력 이미지에 대해, 거대한 3D voxel grid를 생성하므로,
            
            - 저장 공간 효율성이 떨어짐
            - 3D 볼륨 간 interpolation 시, 시각적 오류가 발생할 수 있음!
            - 특수한 상황(앞에서 바라본, forward인 상황)에서만 수행하는 것.
    
    1. NV(Neural Volume)
        
        ⇒ 동적 객체나 장면을 표현하고 렌더링하기 위해 신경망을 기반으로 학습된 3D volume 표현 방식.
        
        여러 시점의 2D img를 입력받고, RGB & Occupancy를 포함하는 3D공간 정보를 학습.
        
        렌더링시, 학습된 volume data에 differentiable인 광선 추적(ray-marching) 또는 volume rendering 기법을 적용하여 새로운 시점의 사실적 이미지 생성.
        
        - 단점
            - 높은 계산 비용(고성능 GPU) 및 느린 학습/렌더링 속도
            - 대량의 학습 데이터 필요
            - 관심있는 대상과 별개로, 배경에 대한 정보를 추가적으로 필요로 함.

---

# #3 Positional Encoding

![스크린샷 2025-11-12 오전 9.52.45.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/05e7999d-4f1f-444a-95b2-9ae8c7c09575.png)

단순 location 값만 FCN에 넣어서 돌리게 되면, 중간 그림처럼 나오게 됨.

positional encoding을 사용하게 되면, 맨 오른쪽과 같이 high frequency를 표현할 수도 있게 됨.

**그걸 어떻게 하는가?**

⇒ manner : low dim에 있는 input을 더 고차원 space로 올리는데, 이때 high frequency function들을 사용하게 됨.

그렇게 나온 결과를 MLP에 넣는 방식.

**그럼 high frequency function의 수식은 어떻게 되는가?**

⇒ 저자들은 heuristically 이 수식을 발견했다고 함.

$$
\gamma(p)=(\sin(2^0\pi p),\cos(2^0\pi p),~...~,\sin(2^{L-1}\pi p,\cos(2^{L-1}\pi p))
$$

- $\gamma$ : $\mathbb{R}$→ $\mathbb{R}^{2L}$ ⇒ simple한 location coordinate에서 고차원 space로 mapping하는 방식
- location coordinate에 대해서는 L=10을 적용
- viewing direction에 대해서는 L=4를 적용
    
    ⇒ 왜 근데 frequency를 다르게 뒀을까?
    
    ⇒ 왜 location 값에 더 많은 frequency를 줬을까?
    
- $F_{\Theta}=F'_{\Theta}\circ\gamma$
    - $F'_{\Theta}$는 그동안 사용한 MLP

![image.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/image%2015.png)

그래서 MLP input들의 parameter 개수의 의문점이 다소 풀리게 됨.

⇒ location 값에 대해서는  x,y,z 각 3개에 대하여, 20만큼 증강을 시켜서 넣어주게 된 것.

⇒ 근데 어떻게 20? ⇒ L=10 아님?

⇒ $\sin, \cos$각각 이 있으니, 하나에 대해, $2\times10$으로 총 20개임.

⇒ 근데 그럼, 각 viewing direction에 대해서는 $\theta, \phi$에 대해서는, $2\times4$,

총 16개만 나올 수 있는데, $\gamma(d)$를 24라고 표현하고 있다.

대체 8개의 coordinate들은 어디서 나왔나?

⇒ 사실 viewing direction을 사용할 때, 이 사람들이 **location coordinate의 Cartesian coordinate vector를 사용했다**고 한다.

![해당 논문 속 일부](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.04.56.png)

해당 논문 속 일부

- **What is Cartesian coordinate system?**
    
    ⇒ 데카르트 좌표계 or 직교 좌표계라고 불리며, 공간 내의 한 점의 위치를 고유하게 결정하기 위해, 서로 직교하는 축을 기준으로 사용하는 좌표체계.
    
    ⇒ x-y좌표계도 여기에 포함되나,
    
    이 논문에서는 x-y-z 좌표계를 차용 ⇒ 3 dimension으로 생각했다는 것.
    
    cf) **What is Cartesian product?**
    
    두 집합의 모든 원소를 가능한 모든 순서쌍으로 결합하는 수학적 개념.
    
    ex) $A = \{a,b\},~~B=\{1,2\}$인 상황에서, 
    
    $A\times B=\{(a,1),(a,2),(b,1),(b,2)\}$로 구성됨!
    

그렇다면, 이제 viewing direction도 별도의 무언가가 아닌, 그냥 location으로부터 도출된 하나의 값이 돼버림. 그래서 24 dimension으로 증강된 것.

- 이후에 저자들이 “**Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains”**라는 논문을 작성
    
    https://arxiv.org/pdf/2006.10739
    
    ![스크린샷 2025-11-13 오후 2.17.26.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.17.26.png)
    

**결국, NeRF Training 과정에서 Positional Encoding을 MLP로 올리기 전에 시행하는 것을 추가로 넣음!**

---

# #4 Results

## #4.1 Qualititive Results

![스크린샷 2025-11-13 오후 2.26.53.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.26.53.png)

![스크린샷 2025-11-13 오후 2.28.07.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.28.07.png)

#2.2에서 언급했던 것처럼, viewing direction에 따라 반사율도 반영하는, non-lambertian 문제도 잘 해결하는 모습을 보여준다.

![스크린샷 2025-11-13 오후 2.30.09.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.30.09.png)

![스크린샷 2025-11-13 오후 2.30.47.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.30.47.png)

high frequency 문제도 잘 해결해주는 모습을 보여준다.

## #4.2 Quantitive Results

![스크린샷 2025-11-13 오후 2.32.31.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.32.31.png)

- What is PSNR?
    
    ⇒ Peak Signal-to-noise ratio
    
    ⇒ 신호가 가질 수 있는 최대 전력에 대한 잡음의 전력을 나타냄.
    
    $$
    PSNR = 10\cdot log_{10}(MAX^2_I/MSE)
    $$
    
    - $MAX_I$ : 해당 영상의 최대 전력값
    - $MSE$ : 평균 제곱 오차
    
- What is SSIM?
    
    ⇒ Structural Similarity Index Measure
    
    ⇒ 참조되는, 왜곡이나 압축이 없는, img에 대해 이미지 quality의 유사도를 측정하는 것.
    
    $$
    SSIM(x,y) = \cfrac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu^2_x+\mu^2_y+c_1)(\sigma_x^2+\sigma_y^2+c_2)}
    $$
    
    - $\mu_X$ : the pixel sample mean of X
    - $\sigma_X^2$ : the sample variance of X
    - $\sigma_{XY}$ : the sample covariance of X and Y
    - $c_i=(k_iL)^2 ~~ for ~~~i=1,2$
    - $L$ : piexel-value들의 dynamic한 range(typically, this is $2^{\#bits ~per~pixel}-1$)
    - $k_1 = 0.01~~and~~k_2=0.03~~by~~default$
    
- **What is LPIPS? + 사람들의 “인식”과 “reconstruction이 잘 됐다”는 것이 allign이 잘 되는가?**
    
    ![위와 같은 방식으로 작동한다고 함. 나중에 관련 논문을 다시 봐야겠음.
    [https://arxiv.org/pdf/1801.03924](https://arxiv.org/pdf/1801.03924)](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.00.30.png)
    
    위와 같은 방식으로 작동한다고 함. 나중에 관련 논문을 다시 봐야겠음.
    [https://arxiv.org/pdf/1801.03924](https://arxiv.org/pdf/1801.03924)
    
    ![스크린샷 2025-11-13 오후 5.01.30.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.01.30.png)
    
    ⇒ Learned Perceptual Image Patch Similarity
    
    ⇒ 비교할 2개의 이미지를 각각 VGG Network에 넣고, 중간 layer의 feature값들을 각각 뽑아내서, 2개의 feature가 유사한 지를 측정하여 평가지표로 사용하는 것.
    
    - What is VGG Network?
        
        ![위와 같은 구조를 사용한다고 함.
        FC : Fully connected layer ⇒ ex) MLP
        Conv : Convolutional layer](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.55.52.png)
        
        위와 같은 구조를 사용한다고 함.
        FC : Fully connected layer ⇒ ex) MLP
        Conv : Convolutional layer
        
        Oxford Univ.의 VGG(Visual Geometry Group)에서 만든 CNNs(convolutional neural networks).
        
        VGG16, VGG19가 있으며, 각각은 마지막에 3개의 FCN을 두고, 그 전까지 16-3, 19-3개의 Conv를 둔 NN이다.
        
        2014 이미지넷 이미지 인식 대회에서 준우승을 한 모델.
        
        https://en.wikipedia.org/wiki/VGGNet
        
    
    아래는 LPIPS가 나오게 된 배경, 실제 사람들이 잘 됐다고 인식하는 거랑, reconstuction이 잘 된거랑 간극이 있었음. 그래서 그 간극을 해소하기 위해 나온 것이 LPIPS!
    
    ![스크린샷 2025-11-13 오후 5.12.03.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.12.03.png)
    
- SRN, NV, LLFF에 대해서는 [여기를 참고](https://www.notion.so/NeRF-Representing-Scenes-as-Neural-Radiance-Fields-for-View-Synthesis-2a6ff2baaf928014aa85eb9696f5fe16?pvs=21)!

- **LLFF가 forward face인 상황에 특히 강점을 보이는 것이라서, 맨 마지막 검사 항목에서 NeRF를 이기기도 함**
- LPIPS는 값이 낮을 수록, 그 간극이 작다는 의미
- 아래는 method 별로 나타낸 것.

![스크린샷 2025-11-13 오후 5.19.55.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.19.55.png)

![스크린샷 2025-11-13 오후 5.20.11.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.20.11.png)

## #4.3 Ablation Study

⇒ 여러 항목들을 변경/제거해가며 성능 테스트한 것.

![스크린샷 2025-11-13 오후 5.26.26.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.26.26.png)

![스크린샷 2025-11-13 오후 5.38.06.png](NeRF%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-13_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.38.06.png)

- 1의 경우, viewing direction도, hierarchical volume sampling도 없이 그냥 해버림. positional encodig도 안 함.
    
    제일 raw한 것.
    
    ⇒ 그래서 성능이 안 좋음
    
- 3의 경우, viewing direction은 없으나, hierarchical volume sampling은 시행. Positional encoding은 시행 X. 그래서 수치가 더 좋아짐.
- 신기한 점은 7,8,9의 frequency에 대한 부분!
    - 마냥 L을 올려도 좋은 게 아니라는 결과가 나옴.
    - 저자들은 이것을 다음과 같이 생각한다고 함.
        
        ⇒ 현재 sample image에 있는 최대 frequency를 $2^L$이 넘기지 않아야, $L$을 늘려서 보강하는 것이 유의미한 결과를 낳는다!
        

## #4.4 Memory&Time Efficiency

- LLFF vs NeRF
    1. For Time efficiency
        - LLFF
            
            ⇒ scene **하나당 10min under**로 3D voxel grid를 뽑아낼 수 있음.
            
        - NeRF
            
            ⇒ single NVIDIA V100 GPU로 scene 하나를 처리하는 데에 **at least 12시간이** 걸렸다고 함.
            
        
    2. For Memory efficiency
        - LLFF
            
            ⇒ scene **하나당 over 15GB**를 사용했다고 함.
            
        - NeRF
            
            ⇒ scene 하나가 아닌! **NN의 weight를 보관**만 하면 됐고, 이를 보관하는 데에 **5MB**가 사용됐다고 함. 이 용량은 single input img 하나의 용량보다 작음.