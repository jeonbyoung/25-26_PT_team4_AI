# Learned Initializations for Optimizing Coordinate-Based Neural Representations

https://www.matthewtancik.com/learnit

paper : https://arxiv.org/pdf/2012.02189

- 여러 개념들에 대한 소개
    - [hypernetwork](https://www.notion.so/Learned-Initializations-for-Optimizing-Coordinate-Based-Neural-Representations-2aaff2baaf928019b94bddd66c369d68?pvs=21)
    - [RNN](https://www.notion.so/Learned-Initializations-for-Optimizing-Coordinate-Based-Neural-Representations-2aaff2baaf928019b94bddd66c369d68?pvs=21)
    - [optimization-based-meta-learning algorithm examples(MAML, Reptile)](https://www.notion.so/Learned-Initializations-for-Optimizing-Coordinate-Based-Neural-Representations-2aaff2baaf928019b94bddd66c369d68?pvs=21)
        - [CNN](https://www.notion.so/Learned-Initializations-for-Optimizing-Coordinate-Based-Neural-Representations-2aaff2baaf928019b94bddd66c369d68?pvs=21)
        - [few-shot training](https://www.notion.so/Learned-Initializations-for-Optimizing-Coordinate-Based-Neural-Representations-2aaff2baaf928019b94bddd66c369d68?pvs=21)
    - [spectral bias](https://www.notion.so/Learned-Initializations-for-Optimizing-Coordinate-Based-Neural-Representations-2aaff2baaf928019b94bddd66c369d68?pvs=21)
    - [Adam](https://www.notion.so/Learned-Initializations-for-Optimizing-Coordinate-Based-Neural-Representations-2aaff2baaf928019b94bddd66c369d68?pvs=21)
- 나중에 공부해볼 것들
    - LSTM
    - MetaSDF
    - DeepSDF
    - Unrolled graident step ⇒ Algorithm Unrolling or Unfolding

# #Overview skimming

⇒ 결국 초기값을 빠르게 설정시켜서 성능을 높여준다는 얘기인 듯.

⇒ 근데 어떻게 빠르게?

⇒ 같은 class(의자, 얼굴, 가면 등) 하나의 통용될 수 있는 $\theta^*_0$를 설정해서 그걸 기준으로 각 이미지에 대한 weight들을 찾는 방식.

⇒ 그리고 그 최적의 초기 $\theta$를 찾는 내용.

⇒ 저자들은 CT 이미지에 대해 이것의 이점을 찾았다고 함.

# #1 Introductions

![스크린샷 2025-11-16 오후 8.39.54.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-16_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.39.54.png)

- MLP를 통해서 2D coordinate을 통해 RGB를 구하려고 할 때, 이 값들이 discrete하지 않고, 연속적이라,(결국 MLP의 weight에 의존할 테니까) 고정된 공간 해상도에 얽매이지 X
    
    ⇒ 그러면, voxel이 주는 3D Complexity(해상도를 높이려 $100\times100\times100$짜리를 $200\times200\times200$으로 올린다고 할 때, $2^3$만큼 공간을 더 늘려야됨. ⇒ 이것이 **cubic complexity**)로 인한 공간 부담도 줄일 수가 있게 됨.
    
- 논문의 저자들은 network weight인 $\theta$가 특정 object에만 최적으로 만들어지는 것을 알게 됨.
- 최적의 $\theta$를 찾는 데에 초기 비용이 크기에, 저자가 제안한 것은 **meta-learning!**
    - 비슷한 것들의 초기 가중치를 잡겠다는 의미인 듯.

근데 문제가 생긴 것은

1. 우선 시간!
    
    ⇒ 기존에 이미 알고 있다시피, NeRF를 traning 하는 데에는 꽤 오랜 시간이 걸린다. 아무래도 high-resolution radiance field다 보니. 
    
    ⇒ 그래서 이걸 조금이라도 줄여보고자, 
    
    - latent(잠재된) vector를 input coordiante에 concatenate(사슬처럼 엮다)해서 single NN이 잘 관리할 수 있게 해줌.
    - 혹은 hypernetwork를 통해 signal 관찰들을 모두 MLP weight로 mapping해줌.
        - what is hypernetwork?
            
            ![이런 느낌인 듯. 좀 더 간소화 시켜서 넘겨주는.](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-16_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.18.00.png)
            
            이런 느낌인 듯. 좀 더 간소화 시켜서 넘겨주는.
            
            기존 NN에서 가중치를 update한다고 하면, 자신의 것을 update하는 것이 일반적이였지만, 이것은 다름. 타겟 network의 가중치를 update해줌.
            
            - meta-learning의 한 형태이며,
            - 입력이나 작업 ID에 따라 타겟 네트워크의 가중치가 동적으로 생성되어, 다양한 작업에 유연하게 대응할 수 있다고 함.
            - 타겟 네트워크의 모든 가중치들을 직접 학습할 필요없어, 효율적이며, 그 크기또한 작다.
            
            ⇒ 약간 연쇄적인 작용으로 parameter update를 하는 것이 부담스럽고,
            
            back propagation이 부담스럽지는 않겠지. 그건 연쇄작용이니까.
            
            ⇒ 근데 fine tuning. 새로운 정보가 들어왔을 때, 그거에 맞게 거대한 network를 재학습 시키는 것도 부담일 터.
            
            ⇒ 더군다나 모든 가중치를 하나에 저장하는 것도 부담일 터.
            
            ⇒ 다양한 이미지가 들어올 때 그에 맞게 처리할 필요가 있어 고안된 방식인 듯.
            
    
2. 그리고 unseen target signal을 표현하기가 힘든 것!
    
    ⇒ 위 두 가지 방법도 결국엔 이 문제를 해결하는 데는 어려움을 보임.
    

for 시간 문제 해결

최근에 연구들에서 meta-learning을 베이스로한 optimization에서 GD step을 굉장히 낮춘 것을 알려줌.

즉, optimization을 통해 2D field를 3D로 바꾸는 것이 필요함.

그런 점에서 저자들은 다시 weight initialization을 다양한 signal type(images, volumetric data, 그리고 3D scene)에서 사용될 것을 제안.

⇒ 초기 network에서 얻게 된 값을 바탕으로 진행하게 된다면, faster convergence도 가능하고, generalizaiton도 더 잘 될 거라고 함. target signal에 대한 view가 부족하더라도.

⇒ 그래서 저자들이 말하는 건, optimization-based-meta-learning algorithm을 이용해서 특정 signal class(face img에 대한 것인 celebA, 3D chair 이미지에 대한 것인 ShapeNet 등)의 초기 가중치를 뽑아내겠다!

- optimization-based-meta-learning algorithm examples(MAML, Reptile)
    1. MAML
        
        Model-Agnostic Meta-Learning(모델 불가지론적 메타 학습)
        
        ⇒ 새로운 작업에 **매우 적은 양의 데이터와 적은 학습 단계만으로 빠르게 적응할 수 있는 모델**을 만드는 것을 목표로 하는 meta-learning algorithm
        
        ⇒ 특정 모델에 구애받지 않고, GD를 사용하는 모든 모델에 적응 가능
        
        ⇒ 2차 도함수도 사용함.
        
        - FOMAML : MAML의 2차 도함수에 대한 의존이 계산 비용도 많이 들고, 많은 메모리가 필요하다고 함.
            
            ⇒ 1차 도함수만 사용하여, meta-learning의 optimization 과정을 단순화 시킴.
            
    2. Reptile
        
        ⇒ MAML의 정교함과 FOMAML의 단순함 사이의 중간 지점을 제시. 1차 도함수를 사용하며, update 방식에 대해, FOMAML을 따라가지 않고, 고유한 방식을 가짐.
        

⇒ 그래서 본인들의 방식의 장점은 simlplicity라고 함. MAML이나 Reptile update 방식으로 수행하기에.

⇒ 그래서 굉장히 단순한 변화일 뿐이기는 하지만, 그 영향은 클 것이라고 함.

# #2 Related Work

## #2.1 Neural Representations

Neural Representations는 최근에 **3D shapes(MLP로 정의된 함수!)**에 관해 최적으로 표현했다고 주목을 받음.

shape의 표면 : 함수를 만족하는 모든 점의 집합!

⇒ 원래 방식은 point cloud에서 3D shape을 만드는 거였고, 나중에는 2D img로 부터도 만들 수 있게 됨.(8D → 5D inputs)

최근 연구에서 ReLU를 activation으로 사용했을 때, spectral bias를 일으켜, fine(미세한) detail들을 처리하지 못 한다.

- **what is spectral bias**
    
    low frequency func를 high frequency func보다 더 쉽게 학습하는 경향
    
    - what is low freq
        
        데이터에서 느리게 변하는 구성요소들. ⇒ 전반적인 평균값을 나타내게 됨.
        
    - what is high freq
        
        객체 가장자리, 질감, 노이즈 등 픽셀 값의 변화가 급격한 부분
        

⇒ 즉, 급격한 색변화를 인지하지 못 하고, 색의 구분이 희미해진다는 뜻인듯.

⇒ 그래서 fourier space에 넣거나, $\sin$ func에 넣어서 처리한다고 함. 이렇게 되면 더 적은 step으로 처리할 수 있다고.

## #2.2 Meat-learning

⇒ 전형적인 few-shot learning 방식

- what is few-shot learning?
    
    https://www.ibm.com/kr-ko/think/topics/few-shot-learning
    
    통칭 FSL
    
    ⇒ 적은 수의 예시만으로도 효율적으로 학습해서 새로운 일을 해결할 수 있는 것을 말함.
    
    ⇒ 사람들이 적은 경험만으로도 새로운 일을 배우고 처리하는 능력을 모방.
    
    ⇒ 말이 쉽지 어떻게 함? ⇒ 전이 학습 or 메타학습 or 둘의 조합
    
    1. **전이 학습**
        
        ⇒ ‘**전이’**라는 단어에서 유추할 수 있듯이, 모델이 이미 학습한 class와 유사한 데이터가 들어왔을 때, fine-tuning을 거치면, 그에 대한 분류도 하게 해주는 것.
        
        - 이때 문서에서는 어떤 class에 대해 학습을 완료했을 때, 출력층의 분류는 바꾸고, 나머지는 동결(더 이상 변하지 않음)한다고 나와있음.
            
            ⇒ 예를 들어, 사자, 호랑이, 얼룩말을 분류하는 기존 모델에서, 입력은 그대로 받되(질감, 형태 등) 분류만 고양이, 강아지로 만든다고 하자.
            
            ⇒ 그럼 그동안 동물들 분류하면서 만든 가중치들은 유지되고, 최종 출력층의 가중치만 바꾸면 되기에, 빠르고 효과적으로 학습이 가능하다고 한다.
            
    2. **meta-learning**
        
        지도 학습 or fine tuning : train에 모델이 test할 class도 다 들어가 있는 방식
        
        ⇒ meta-learning : 조금 더 광범위하고 간접적인 접근 방식을 취함.
        
        ⇒ 전이 학습과도 궤를 달리함.
        
        ⇒ 전이 학습이 기학습된 것에서 살짝의 tuning을 하는 거라면, meta-learning은 시스템을 E2E로 학습함.
        
        ⇒ 여러 훈련 에피소드에 걸쳐 모델 함수를 훈련시킨 뒤, 아직 보지 못한 class를 포함해,
        
        ⇒ class의 데이터 요소 간의 유사성에 대한 예측을 출력하고,
        
        ⇒ 이번에 배운 내용을 바탕으로 다운스트림(구체적으로 정의된 분류 문제)를 해결.
        
        **What??????**
        
        ⇒ 메타 인지라는 말 들어봤듯이, 그냥 **학습하는 것들을 학습하기** 라고 생각!
        
        ex) 언어 시험을 본다고 가정하자.
        
        - 딥러닝 :
            
            불어 문제 1000개를 주고 train시킨 뒤, 스페인어 문제로 test를 함
            
            ⇒ 특정 문제에 특화돼있지만, 다른 걸 주면 해결을 못 함.
            
        - meta-learning :
            
            **불어/한국어/영어/일본어/독일어/중국어/… 각 5개씩 주고(여러 훈련 에피소드 들) train** 시킴, 그리고 test는 스페인어로 봄.
            
            ⇒ 전반적인, 여러 문제들을 train하며, 이들의 유사성인 **언어**는 이런 거구나(**class 데이터 요소 간의 유사성 학습)** 하고 알게 됨.
            
            ⇒ 그에 대한 **요령**으로 **스페인어(다운스트림)** 문제를 풀 수 있다고 함.
            
        
        ⇒ 이번 논문으로 전환시키면, 여러 3D object를 주면, 그들의 형상은 모두 달라도, 그 3D object들의 특징을 배우게 될 것임
        
    
    - 아래는 FSL의 대표적인 아키텍처인 N-way- K-shot 방식을 나타낸 것.
        
        ![examples of 3-way-2-shot](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.08.40.png)
        
        examples of 3-way-2-shot
        
        - N : # of class
        - K : # of sample
- What is CNN?
    
    https://wikidocs.net/64066
    
    Convolutional Neural Network(합성곱 신경망)
    
    ⇒ 인간의 시신경을 최대한 본 따 만든 것.
    
    먼저 용어 정리를 좀 해야됨
    
    ![스크린샷 2025-11-17 오후 8.49.13.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.49.13.png)
    
    모두 3차원 tensor 기준
    
    - 높이 $I_h$
        
        ⇒ 3차원 tensor가 주어졌을 때, 세로로 열이 이뤄진 것들
        
    - 너비 $I_w$
        
        ⇒ 3차원 tensor가 주어졌을 때, 가로로 열을 이룬 것들
        
    - 채널 $C_i$
        
        ⇒ 그렇게 $너비 \times높이$ 로 주어진 것들이 뒤로 겹겹이 쌓인 것들
        
        ex) color img의 경우 : rgb로 채널 값이 3
        
        흑백 img의 경우 : 0(백) 1(흑)이니까 1 로 할 수 있음.
        
    
    ![l : 입력, r : kernel](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.51.12.png)
    
    l : 입력, r : kernel
    
    - 커널 kernel (or filter)
        
        ⇒ $너비\times높이\times채널$로 구성된 얘들을 커널에 맞춰 감싸고, 입력 데이터의 각 성분에 맞는 값들을, 커널의 각 성분에 맞게 곱하는 그것
        
    - 스트라이드
        
        ⇒ 커널 크기만큼 입력 데이터를 감쌀 때, 한 번 감싸고 다음으로 이동해야하지 않음? 
        
        ⇒ 그 이동하는 단위 설정하는 것.
        
    - 합성곱 연산
        
        ⇒ 그냥 입력 데이터를 커널로 감싼 것과 **짝이 맞게 곱한** 것!
        
    - 패딩(padding)
        
        ![스크린샷 2025-11-17 오후 9.09.00.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.09.00.png)
        
        ⇒ 합성곱 연산을 시행하면, 필연적으로, 그 크기가 줄어들 수 밖에 없음.
        
        ⇒ 그럼 애초에 입력을 좀 크게 잡아놓으면 안 줄겠다!
        
        스트라이드 값 1 기준,
        
        ⇒ 만약 kernel 크기가 3by3이였다면, 1폭짜리 zero padding을.
        
        ⇒ 만약 kernel 크기가 5by5였다면, 2폭짜리 zero padding을.
        
        두게 되면 패딩 전의 크기와 똑같을 것임!
        
    
    - 합성곱 층
        
        ⇒ 입력과 커널로 하나씩 곱한 뒤, 그 결과들에 activation(generally, ReLU or revised ReLU)를 시행한 것.
        
    - 풀링
        
        일반적으로 출력층에서 주로 쓰이며, 커널로 감싸는 건 하는데, 그 감싼 값들 중에
        
        1. 최대 풀링(max pooling) : 가장 큰 값을 출력하는 것.
            
            ![스크린샷 2025-11-17 오후 8.55.51.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.55.51.png)
            
        2. 평균 풀링(average pooling) : 그 커널에 들어온 얘들 중에 평균값을 출력하는 것.
    
    **🚩 시작!**
    
    ⇒ 애초에 CNN은 왜 나왔는가?
    
    ![스크린샷 2025-11-17 오후 8.56.51.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.56.51.png)
    
    MLP가 이런 거 분류에 어려움을 느낌.
    
    ⇒ **지정된 위치에 있지 않고,** 다르게 생겼으니까.
    
    ⇒ 근데 인간은 이게 좀 찌그러져서 그렇지 둘이 같이 분류돼야되는 걸 앎!
    
    ⇒ 어떻게? ⇒ 하나씩이 아니라, **범위단위로 인식**하니까.
    
    ⇒ 그럼 컴퓨터도 범위로 인식하게 해야겠다!
    
    ⇒ 그 것이 kernel
    
    ![스크린샷 2025-11-17 오후 9.02.34.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.02.34.png)
    
    ![스크린샷 2025-11-17 오후 9.02.54.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.02.54.png)
    
    다음과 같이 $5\times5$라는 입력 데이터가 있으면, kernel이 이 값을 각 성분 별로 곱하여, 새로운 출력을 만들어내게 됨.
    
    ⇒ 주로 kernel은 $3\times3~~~or ~~~5\times5$의 크기를 가짐!
    
    그렇게 9번을 반복하면, 아래와 같은 feature map을 만날 수 있음.
    
    ![스크린샷 2025-11-17 오후 9.10.35.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.10.35.png)
    
    위 예시에서 볼 수 있듯이, 2번째 스텝에서 한 칸만 옮겨감 ⇒ 스트라이드 : 1
    
    ⇒ 그럼 스트라이드 2인 경우는?
    
    ![스크린샷 2025-11-17 오후 9.04.47.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.04.47.png)
    
    위에서 볼 수 있듯이, 출력인 feature map은 그 size가 줄어드니, 그게 맘에 안 들면 padding하는거지.
    
    **근데 여기서 드는 생각!**
    
    ⇒ 근데 이거 사실 생각해보면, kernel의 각 값이 그냥 가중치처럼 작용한 거 아닌가?
    
    ![스크린샷 2025-11-17 오후 9.12.12.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.12.12.png)
    
    이게 지금 일어난 상황.
    
    ⇒ 그래서 MLP를 FCN(fully connected network)라고 부르는 이유를 알게 됨. MLP는 저걸 모두 다 이은 거니까.
    
    ⇒ 여기서 보여지듯, CNN이 특징은 FCN보다 적은 가중치를 이용하면서, 공간적 구조를 보존해주는 특징이 있음.
    
    ⇒ 근데 그럼 MLP 처럼 bias는 못 두나? ⇒ 둘 수 있음!
    
    ![스크린샷 2025-11-17 오후 9.14.54.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.14.54.png)
    
    **위와 같이 feature map에 편향 값을 다 더해주면 됨!**
    
    그럼 kernel 값도 정해져. input값도 우리가 알고 있어. 그럼 우리는 feature map의 size도 뽑아낼 수 있음.
    
    ⇒ 근데 왜 뽑아냄?
    ⇒ feature map의 size가 결국, kernel을 몇 번 곱해야하는 지 알려주는 거니까
    
    ![스크린샷 2025-11-17 오후 9.17.46.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.17.46.png)
    
    위와 같을 때, + $P$ : 패딩 폭 ⇒ 아래서 $2P$인 이유는 양옆/상하 로 붙여주니까
    
    $O_h = floor(\frac{I_h-K_h+2P}{S}+1)$,    $O_w=floor(\frac{I_w-K_h+2P}{S}+1)$
    
    왜 floor을 씌웠냐? ⇒ 소숫점을 무시하기 위함.
    
    1은 왜 더했나? ⇒ 마지막으로 갈 수 있는 것까지 더해주려고.
    
    여기까진 채널이 1이였을 때였음!
    
    ---
    
    **이제 채널이 2개 이상일 때를 봐보자.**
    
    ⇒ 추가되는 건 그냥 kernel의 채널에 대한 부분!
    
    아래와 같이 수행**할 수**도 있음!
    
    ![스크린샷 2025-11-17 오후 9.26.24.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.26.24.png)
    
    입력 채널이 여러 개이면, 그에 맞게 kernel도 여러 개여야됨!
    
    ⇒ **즉, 입력과 kernel의 채널이 같아야 됨!**
    
    ⇒ 여기서 착각하지 말아야할 것은 kernel은 여러 개가 아님!
    
    ⇒ kernel은 1개! 그 kernel의 채널이 여러 개인 것!
    
    ⇒ 근데 이 경우에는 RGB 값이 하나로 합쳐지게 됨!
    
    **⇒ 이걸 추상화된다고 표현하기도 하는 듯 ⇒ 어쨋든 압축시키는 방식인 것임.**
    
    3차원 tensor의 경우도 봐보자.
    
    ⇒ 그리고 **kernel을 여러 개 쓰는 상황**이라고 가정해보자!
    
    ⇒ 이때 쓰이는 매개변수의 총 개수도 따져보자. input의 채널이 $C_i$라 할 때,
    
    ![스크린샷 2025-11-17 오후 9.40.55.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.40.55.png)
    
    ⇒ $K_h\times K_w \times C_i \times C_O$
    

⇒ 새로운 instance가 test time에 들어와도, 빠르게 최적화시키는 gradient를 base로 하는 최적화 방식으로, SGD나 Adam 등, 기존 구현 위에서 쉽게 처리 가능 

- What is Adam?
    
    Adaptive Moment Estimation
    
    ⇒ SGD에 모멘텀 및 RMSProp을 결합해서, GD의 효율성을 크게 향상시킨 것.
    
    - What is momentum? simply
        
        단어 뜻 그대로 **관성**을 의미!
        
        이전에 시행되던 거를 이어서 한다는 느낌
        
        $$
        v_t = \gamma v_{t-1}+\eta \nabla_{\theta_t}J(\theta_t)\\ \theta_{t+1}=\theta_t-v_t
        $$
        
        위 방식으로, parameter update를 수행
        
    - What is RMSProp? simply
        
        AdaGrad라는 게 있는데 그것의 문제가 vanishing gradient였다고 함.
        
        그 문제를 해결해주려고 만들어진 것.
        
        ⇒ 결국 핵심 기능은 learning rate를 상황을 봐가며 조절해주는 것.
        
    
    ⇒ 학습 속도 조절이 가능해서, train 초기-후반에서의 학습률 변화에 따라 유연한 조정이 가능하다.
    

⇒ 그럼 기존엔 어떠했나?

⇒ LSTM과 같은 meta-learner(선생님 역할)을 둬서, test가 끝나면 그에 따른 learning rate와 같은 hyper-parameter(train과 무관하게 설정돼있는 얘들, parameter update 방식도 들어가있겠지)를 조정했음.

- What is LSTM?
    
    RNN의 한 종류로, RNN의 장기 의존성 문제를 해결하기 위해 나온 모델
    
- **What is RNN?**
    
    ref) https://www.youtube.com/watch?v=OkTyY28XMuQ
    
    Recurrent Neural Network
    
    ⇒ 순서 나타나는 데이터들을 처리하는 데에 유용한 것. ⇒ 약간 논리회로에서 flip-flop이랑 비슷한 것 같음.
    
    ⇒ ‘순서’에는 언어가 들어갈 수도, 숫자가 들어갈 수도 있음!
    
    ex) ___ bat라는 것을 봤을 때, bat는 다음 2가지로 해석될 여지가 있음!
    
    - 야구 방망이
    - 박쥐
    
    근데 앞에 빈칸이 baseball이였다면? ⇒ 야구 방망이가 뜻일 확률이 높아질 것임.
    
    즉, **문맥**을 고려할 수 있게 되는 것! ⇒ 인간의 뇌 역할을 해주게 됨.
    
    ![스크린샷 2025-11-18 오전 9.25.37.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_9.25.37.png)
    
    다음과 같은 상황에서, $x$는 bat를, $h$는 baseball을 의미하며, $W,U$는 영어→한국어 번역시켜주는 것을 의미한다. $W$를 거친 $h$는 $x$의 뜻이 결정될 확률을 결정해줌.
    
    이걸 조금 더 디테일하게 풀어갈 건데, one-hot-vector 방식을 사용할 것임.
    
    ![스크린샷 2025-11-18 오전 9.31.27.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_9.31.27.png)
    
    다음과 같은 구성을 해주고,
    
    이를 위 RNN에 넣고 forwarding 시키며, 시간 순서에 따른 변화를 살펴보자!
    
    먼저, h는 다음과 같이 쓸 수 있음. $h_1 = tanh(U_x + Wh_0+b_0)$.
    
    다음으로, o와 $\hat{y_1}$은 다음과 같이 쓸 수 있음. 
    
    $o_1 = Vh_1+b_1\\\hat{y_1} = softmax(o_1)$
    
    근데 이제 bias는 계산의 편의를 위해 제외한다고 함.
    
    초기값 설정은 다음과 같이 했음. $h_0$는 원래 없는 값이니, 0으로 구성.
    
    ![스크린샷 2025-11-18 오전 9.36.36.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_9.36.36.png)
    
    이를 위 식에 맞춰서 넣어줄 건데, baseball이 먼저 나오니, 그것 먼저 넣어줄 것임.
    
    $h_1 = tanh(\begin{bmatrix} {0.094} & {-0.02} & {0.135} \\ {0.135} & {-0.069} & {-0.009} \end{bmatrix}\begin{bmatrix} {1}\\{0}\\{0}\end{bmatrix}+\begin{bmatrix} {-0.011} & {0.13} \\ {-0.123} & {0.014}\end{bmatrix}\begin{bmatrix} {0}\\{0}\end{bmatrix})\\~~~~~=tanh(\begin{bmatrix}{0.094}\\{0.135}\end{bmatrix})=\begin{bmatrix}{0.094}\\{0.135}\end{bmatrix}$
    
    근데 어떻게 tanh라는 함수를 거쳤는데, 값이 거의 그대로 나온걸까?
    
    ⇒ **tanh는 0 부근에서 y=x와 비슷한 성질**을 띠기 때문
    
    그럼 마찬가지로 나머지 식도 넣어주면 다음과 같이 결과를 가져올 수 있게 됨.
    
    ![스크린샷 2025-11-18 오전 9.55.24.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_9.55.24.png)
    
    마찬가지로, bat에 대해서도 식을 써주게 되면, 다음과 같은 결과를 가져올 수 있음.
    
    $o_2=\begin{bmatrix} {-0.022} \\ {0.01} \\ {-0.022}\end{bmatrix}~~~~~~~~\hat{y_2}=\begin{bmatrix} {0.33} \\ {0.341} \\ {0.33} \end{bmatrix}$
    
    ![스크린샷 2025-11-18 오전 9.56.07.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_9.56.07.png)
    
    이후, cost func로는 CE를 쓰고, backpropagation을 수행하게 됨.
    
    RNN에서 쓰이는 backpropagation은 **BPTT**라고 불리며,
    
    술어는 Back Propagation Through Time이라고 불린다고 한다.
    
    $V = V_1 + V_2$라 할 때, $\partial J/\partial V = \partial J/\partial V_1 ~~+ ~~\partial J/\partial V_2$라고 표현되며, 이를 chain rule 쓰게 되면, $\partial J/\partial V_1 = \frac{\partial J}{\partial o_1} \times \frac{\partial o_1}{\partial V_1}$으로 표현 가능하다.
    
    (어짜피 $V_2$에 대해서도 같을 테니, $V_1$을 기준으로 설명)
    
    이때, J는 CE임을 고려하고, $\hat{y_1}=softmax(o_1)$임을 고려하면,
    
    $\frac{\partial J}{\partial o_1}~=~(\hat{y_1}-y_1)$이 됨. 아래는 그 proof.
    
    ![IMG_07B563A93C11-1.jpeg](Learned%20Initializations%20for%20Optimizing%20Coordinate-/IMG_07B563A93C11-1.jpeg)
    
    상수값으로 표현될 값들은 제외한 듯.
    
    $\frac{\partial o_1} {\partial V_1} = h_1 ~~(\because o_1=V_1h_1)$으로 다음과 같이 표현해줄 수도 있음.
    
    이를 이용해, gradient를 다음과 같이 표현할 수 있고,
    
    ![스크린샷 2025-11-18 오후 1.39.12.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.39.12.png)
    
    실제 계산을 하면 다음과 같음
    
    ![스크린샷 2025-11-18 오후 1.39.32.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.39.32.png)
    
    이제 $\partial J/\partial W$도 알아야됨. (참고한 영상에서는 손실함수 표현을 $L$로 해서 형식이 좀 다름)
    
    그래서 비슷한 방식을 사용하는데, 여기서 **문제는 이것이 순차적인 구조를 띤다는 것**임.
    
    ![스크린샷 2025-11-18 오후 1.50.12.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.50.12.png)
    
    ![스크린샷 2025-11-18 오후 1.49.49.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.49.49.png)
    
    즉, 위와 같이 두 개의 케이스를 고려해서, $\partial J_2/\partial W$의 식을 구성해줘야됨.
    
    아래는 각각이 나온 것이며, 동그라미 표시한 부분이 $h_2$와 $h_1$ 사이의 관계를 표현한 것이다.
    
    ![스크린샷 2025-11-18 오후 1.52.04.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.52.04.png)
    
    이후, 그동안 구했던 방식을 사용하고, activation func을 tanh로 사용했던 점을 고려해서 전체 식 구성을 하면 다음과 같은 풀이가 완성된다.
    
    ![스크린샷 2025-11-18 오후 1.54.46.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.54.46.png)
    
    초기에 세팅을 $h_0$는 0벡터로 세팅해서 뒤는 0이 된다.
    
    ⇒ **근데 실제 RNN을 구현할 때는, 앞선 sequence에 의해 계산된 마지막 $h_{t-1}$로 설정되는 경우가 많음.**
    
    **즉! 순환 구조를 이루게 설계된다는 것!**
    
    U에 대한 편미분 값도 다음과 같이 얻을 수 있다. W를 통해 구할 때와 거의 유사하다.
    
    ![스크린샷 2025-11-18 오후 1.59.09.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.59.09.png)
    
    이런 방식으로 parameter update를 수행할 수 있다.
    

⇒ 다소 복잡함! 추가로 meta-learner도 학습시켜야하니, 부담이 꽤 큼!

**⇒ 즉, 본인들의 방식이 단순하다라는 걸 강조하고 싶었던 듯.**

MetaSDF라는 걸로 이걸 검증해봤는데, 실제로 standard하게 하는 DeepSDF보다 더 빠르게 수렴하고 좋더라~.

- what is MetaSDF?
    
    signed distance functions(부호화 거리 함수)를 메타학습하는 framework로, Stanford에서 만들었다고 함. 나중에 따로 공부…
    
- What is DeepSDF?
    
    signed distance functions를 학습하는 딥러닝 기반 방법. 이것도 나중에 공부…
    

그래서 우리도 그렇게 함 해볼거다!

---

# #3 Overview

- 결국 우리가 하고 싶은 일은 무엇인가?
    - 2D pixel coordinates → 3D의 color value (rgb)를 알아내거나,
    - 3D shape(voxel 같은 거 말하는 듯) → 4D tuples(color(RGB) & density를 담은)를 찾기
    

우리가 원하는 값의 ground truth, 즉, 실제값을 T라 할 때, loss func은 square of $L_2$  norm으로 정의해서 구할 수 있음

![스크린샷 2025-11-18 오후 5.48.01.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.48.01.png)

그리고 parameter update도 기본적으로는, GD에서 했던 방식 그대로 그냥 수행하면 됨.

$$
\theta_{i+1}=\theta_{i}-\alpha \nabla_\theta L(\theta)|_{\theta=\theta_i}
$$

그리고 Adam 같은 optimizer들이 이 parameter 변경되는 궤적? 추이? 를 계속 쫓고 있을 것임.

그리고 정해진 optimization step을 다 수행하고 나온 $\theta_m$에 대해서, 손실함수를 계산해보면, 최종 결과를 구할 수 있을 것이고,

지금 접근하고 있는 문제인 3D reconstruction에 관해서는 다음과 같이 구성해볼 수 있을 것임.

- $M(T,p_i)$ : 영상이 찍힌 상황에서, camera pose p에서 3D object T의 모습을 찍은 2D img.
- $M(f_\theta ,p_i)$ : 신경망의 3D 예측하는 함수인 $f_\theta$로부터 $p_i$ 자세에서 렌더링된 2D img.
    
    ⇒ 웬 2D img? 우리는 3D 복원하려는 거 아니였음?
    
    ⇒ 역추출(solving an inverse problem)하려는 것임.
    
    ⇒ 원래 값의 오차가 작아지게 만들면, 역설적으로 가장 유사한 color/density값도 알 수 있게 될 것임.
    

⇒ 근데 이 방식은 문제가 **관찰 값이 너무 적다면**, $f_\theta$가 T에 가까이 다가가질 못 할 것임.

⇒ 그리고 그 **3D object에 대한 사전 지식 없이는 만들어내기 힘듦!**

이런 문제 상황이 도출했으니, meta-learning을 생각해보게 된 듯.

---

## #3.1 Optimizing initial weights

⇒ 특정한 distribution $\mathcal{T}$(2D face imgs or 3D chairs)에서 나온 signal T가 있다고 생각하자.

⇒ 우리의 목표는 최적의 초기 가중치 $\theta^*_0$를 찾으려고 하는 것이다.

$\theta^*_0=\underset{\theta}{\argmin}E_{T\sim \mathcal{T}}[L(\theta_m(\theta_0,T))]$라는 것을 해결하려고 하는 건데,

위에서 언급했듯, good starting을 위해, meta-learning(MAML, Reptile 등)을 생각해보게 됨.

1. MAML
    
    inner loop와 outer loop라는 게 있음
    
    - inner loop : m optimization steps을 거치며, $\theta_m(\theta_0,T)$를 계산하는 것.
    - outer loop : inner loop 주위에서 초기 가중치 $\theta_0$를 학습하려고 meta-learning을 하는 것.
        
        ⇒ 각각의 outer loop는 $\mathcal{T}$로부터 signal $T_j$를  뽑아낼 수 있고, update 식은 다음과 같음
        
        $\theta^{j+1}_0=\theta^j_0-\beta \nabla_\theta L(\theta_m(\theta,T_j))|_{\theta=\theta_0^j}$
        
        그리고 meta-learning step size는 $\beta$.
        
        ⇒ 근데 좀 이해가 안 되는 부분이 있었음.
        
        ⇒ $L(\theta_m(\theta,T_j))$ 이것에서 정답값은 뭘까?
        
        ⇒ $\theta_m(\theta^j,T_j)$의 의미는 초기 값 $\theta_0^j$를 가지고 태스크 $T_j$를 수행한 뒤의 task $T_j$에 대한 예측 값을 의미하고
        
        ⇒ $L(\theta_m(\theta,T_j))$는 결국, 진짜 task T와 그에 대한 예측을 의미함.
        
        ⇒ 찾아본 바로는, 더 단순화하면,
        
        - **inner loop** : 초기 $\theta$값에서 batch size만큼 task $T_j$에 대해 inner loop를 돌려서 최종 $\theta_m$을 찾는 과정.
        - **outer loop** : 그렇게 inner loop가 끝나면 outer loop의 meta-learning parameter update를 1번 수행함.
        - **what is task?**
            
            ⇒ 그냥 해결해야될 문제 하나라고 생각하면 됨.
            
            ex) 강아지/고양이, 사자/호랑이 분류등 각각이 task가 되는 것임.
            
        
        ⇒ 결국 inner loop에서 가져온 값으로 초기화 시켜주는 것.
        
        ⇒ 근데 생각해보면, $\theta_m(\theta,T_j)$가 이미 $\nabla$를 inner loop에서 거치고 왔으니까, 
        
        사실상 이건 2차 도함수를 계산하는 것이나 마찬가지임.
        
2. Reptile
    
    MAML과 비슷한 방식을 사용하지만, 더 단순한 update rule을 적용한다. 
    
    2차 도함수를 필요로 하지 않는다는 점에서 그렇다.
    
    $\theta^{j+1}_0=\theta^j_0-\beta  (\theta_m(\theta,T_j)-\theta^j_0)$
    
    위 식을 통해서 계산한다.
    
    ⇒ 이 또한 
    
    **잘 구해놓은 $\theta_m$을 가지고, $\theta^j_0$와 비교해서 얼마나 이동했는 지 확인하고 그 방향대로 옮겨주는 것이라고 생각하면 될 듯.**
    

---

## #3.2 Experimental setup

그래서 저자들은 이 두 가지 방식을 다 사용해봄.

- MAML의 경우, 정해진 inner loop step에 대해, 더 나은 성능을 보여줌.
- Reptile의 경우, less memory-intensive여서(그냥 선형결합으로 이뤄져서 인듯) inner loop step을 더 많이 사용하지 않게 됨.

몇몇 경우엔, MAML은 inner loop step이 이거한테는 부족해서, target signal에 대해, 충분한 관찰 결과를 가져오지 못 했다고 함.

그래서 이런 경우엔, Reptile을 사용해서 inner loop 과정동안에 여러 view를 얻어냈다고.

그래서 실험 설정은 다음과 같음

1. Meta-learning에 대해서는, MAML과 Reptile을 여러 물체들을 관찰하는 task들에 사용하고, 초기 가중치를 최적화 시킴
2. Test-time optimization에 대해서는, standard한 GD 기반의 최적화를 사용해서, 같은 class의 이전에 보지 못 했던 view를 보여주는 방식을 사용.

⇒ 저자들의 관심은,

어떻게 test-time 최적화 동안에 다른 여러 초기 가중치 세팅들이 새로운 signal에 fit되게 neural representation에 영향을 미칠까?

⇒ 결국 여러 개 시험해보고, 특정 초기 가중치를 설정하는 게 최적이라는 말을 하려는 듯.

---

# #4 Results

저자들의 실험 내용과 결과는 다음과 같았음.

1. 2D img regression : CT 사진을 사용
    
    ⇒ meta-learned인 가중치 초기화 방식은 적은 수의 view로도, test-time에서 더 빠른 convergence를 보였음.
    
2. 3D shape reconstruction : test-time에는 Phototourism을 사용했다고 함.
    
     https://phototour.cs.washington.edu/datasets/
    
    ⇒ 얘도 test-time 동안에 signle view reconstruction이 가능하고, 더 빠르게 convergence함을 알 수 있었다고 함.
    

---

## #4.1 Tasks

지금부터 할 얘기들은 각 task에서 잡은 basic setup에 대한 얘기임.

### #4.1.1 Image Regression

원래 구조는 MLP와 같았다고 함. pixel coordinate을 주면 RGB 값을 내주는.

여기서 저자들은 $\mathcal{T}$라는 dataset을 4개정도 준비했음

1. CelebA : 얼굴을 담은 이미지들 https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Imagenette : 그냥 이런 저런 이미지들 https://github.com/fastai/imagenette
3. Text : text를 찍은 이미지들 https://www.kaggle.com/datasets/emrahaydemr/handwriting-same-text-image (대충 뭐 이런 거인듯)
4. SDF : 2D signed distance fields of simple curves https://github.com/mintpancake/2d-sdf-net(대충 뭐 이런 거인듯)

각 세트 당 10,000개의 sample을 가져왔으며, $178\times178$ 사이즈의 pixel로 가져옴.

사용한 MLP의 architecture는 5 layer with 256 nodes(channel이라고 표현됐는데, 같은 의미인듯) 이며, sine function nonlinearities를 사용했다고 함.

⇒ activation func로 $\sin$을 사용했다는 뜻임. 미분에 있어서 ReLU가 가지는 안 좋은 점을 보완하려는 것인  듯. SIREN이라는 논문에서 이게 잘 표현된 듯…https://tskim9439.github.io/blog/ai/2022-06-16-SIREN/

### #4.1.2 CT reconstruction

#3에서 언급했던 것처럼, solving an inverse problem, 역추출하려는 것임.

⇒ CT 사진은 단층 촬영술, 그 방식은 1D에서 그 경로상의 밀도를 측정하고, 그 값을 바탕으로 사진을 찍어내는 방식인데, 이를 역이용해서,

CT img → 해당 지점의 density를 도출하는 것.

- 여기서 $\mathcal{T}$는 2048개의 $256\times256$ 사이즈의 Shepp-Logan phantoms(https://en.wikipedia.org/wiki/Shepp%E2%80%93Logan_phantom)라는 dataset으로 구성돼있다고.
    - Shepp-Logan phantoms는 256개의 평행한 ray가 random한 방향에서 지나가며 만들어낸 이미지이고, 이걸 통해서 meta-learn 하겠다는 것.
- Reptile을 meta-learn에 사용해서, 초기 가중치를 구하는 데, 12 unrolled gradient steps 동안에 구한다고 함.
    - What is unrolled graident step? simply
        
        ⇒ deep하게 들어가면 unrolled GAN이라는 게 나오던데, 추후 학습…
        
        ⇒ unrolled(말려있지 않은, 펼쳐진) gradient step(parameter update를 말하는 듯.)
        
        즉, inner loop에서 k번 만큼 돌리고, 각각 돌렸을 때의 가중치를 $\theta_0\sim\theta_{k-1}$이라고 할 때, 특정 시점 $\theta_t$일 때, 이때의 결과가 어떤 지 알려주는 것.
        
        ⇒ 근데 이거 이미 하던 거 아님? back propagation 시에, 그 특정 시점의 $\theta$에 대해 어떤 지를 확인하는 작업을 거치니까.
        
        ⇒ 근데 이제 그동안 한 것은 $\theta$ 자체를 고치는 거였고, 지금 하는 건 meta-learning이니, 
        
        $\theta^*_0$를 구하려고 하는 것! ⇒ 다름!
        
- 근데 MAML을 써서 구하니까, 3번만 사용할 수 있었다고 함. 즉, Reptile방식이 더 좋았다는 거임.
    
    ⇒ MAML은 메모리를 너무 많이 잡아먹어서!
    
- 이때에는 activation func으로는 ReLU를 썼다고 함.
- 그리고 input에는 random한 Fourier feature를 적용시켜서 전처리를 거쳤다고.
    
    ⇒ coordinate base를 frequency space로 embedding 시켰다는 것.
    

### #4.1.3 View Synthesis for ShapeNet objects

3D object에 관해서는 simplified NeRF를 사용하겠다고 했음.

⇒ 근데 기존 NeRF가 viewing direction을 feed했던 것을 시행하지 않는다고 하고, 

⇒ 또한 Coarse and fine으로 시행됐던 Hierarchical Volume Rendering도 시행하지 않고, single 방식으로 구성했다고 함.

- ShapeNet에 있는 object 중, synthesis에 사용될 category는 다음과 같음
    
    ⇒ $\mathcal{T}$ : chair, car, and lamp
    
- 각 task에 대해, 25개의 $128\times128$ 사이즈 img들을 준비했음.
- 그 object들을 바라보는 viewpoint들은 모두 random하며,
- object들은 canonical(표준적인) 규격 안에 놓여지게 됨.
- 각 object들은 random하게 설정된 환경에서 lit(빛을 받다, 조명 받는 거)되고 있고, ray tracing에 의해 rendering될 것임.
- Reptile 사용해서,  32번의 unrolled gradient step 동안에 각 카테고리에 대한 meta-learn을 수행했음.
- MLP의 architecture는 6 layer wi 256 nodes 라고 하며, activation으로는 ReLU를 쓰고,
- input에 positional encoding을 사용했다고 함. ⇒ 기존 NeRF 구조와 동일.

### #4.1.4 View Synthesis for Phototourism scenes

 https://phototour.cs.washington.edu/datasets/

이 데이터셋은 유명 관광소에 대한 사진이 담김.

- 결국 목표는 이런 것들을 학습시켜서, 새로운 view에서 바라본, 조명 컨디션이 다를 때의 모습을 렌더링해내는 것이였음.
- 주요한 문제는, capture를 했을 때마다 달라지는 조건들이였음.
    - 관광지다 보니까, 사람들과 차들처럼 transient(일시적인 것들)도 있을 거고,
    - 조명 조건 등도 달라질 것임.
- 그래서 정답 값(signal)은 특정 조건(시간, 조명, 기상상황 등)에 대한 그 landmark의 모습이 될 것임.
    - landmark에는 Trevi, Sacre Couer, or Brandenburg가 들어간다고.
        
        ⇒ 다 예쁘게 생긴, 관광 명소들임.
        
- 근데 만약에 이런 조건 하에 있는 건데, standard NeRF를 사용하게 된다면, 이런 문제들도 모두 학습을 할거고, 결과적으로 건축물의 변경도 일으킬 것임.
- 그래서 저자들은 이런 문제들도 단지 초기 가중치를 잘 잡았을 뿐인데, 어느 정도는 문제를 해결했다고 함.
- #4.1.3에서 시행됐던 MLP 모델의 architecture가 좋았나봄. 깡통 상태인 그것을 그대로 가져다 썼고, 64번의 unrollled gradient step을 거쳤다고 함.
- meta-learn을 할 때에는 각 랜드마크에 대해 1000장의 사진을 가지고 진행했다고.
- test-time에는 다른 view point로부터 새로운 view를 가져오기도 함.

---

## #4.2 Baselines

저자들은 여러 초기화 scheme들을 자신들의 실험 세팅 값들과 비교해봤다고 함.

1. Mean(평균으로 시작) : 처음부터 network의 예측값을 $\mathcal{T}$에 있는 값들의 평균으로 세팅해놓는 것
2. Matched(결과는 같은 걸로 시작) : 처음부터 meta-learn돼있는 거랑 똑같은 결과물을 내보내게 파라미터를 세팅해놓은 것.
    
    ⇒ 물론 meta-learn된 거랑 weight가 같을 수도 있지만, in general, 다를 수 밖에 없음.
    
3. Shuffled : 현재 class $\mathcal{T}$에 대해**,** meta-learned된 초기값 $\theta^*_0$를 가진 network의 가중치를 각 layer내의 값들과 서로 섞는다.

⇒ 1,2 방법들은 그럼 사실, signal space, 즉, 이 class에 대한 정답과의 loss가 적은, 좋은 시작값을 가지고 있는 것임. 애초에 평균이나, 결과는 같은 걸로 시작한 것이니.

⇒ 근데 막상 이걸 GD를 시켜보면, 즉, 새로운 걸 가져와서, 이거에 대해 얼마만에 예측하나 보면, meta-learn된 거에 비해서 한참 느린 것을 알 수 있음.

**⇒ 즉, signal space보다, weight space가 더 중요하다!**

⇒ 3번의 경우, 최적 초기값을 찾아냈지만, 그것을 shuffling하다보니, 잘 converge하거나, generalization을 수행하는 데에는 좀 아쉬웠다.

그 뒤에 최적화시키는 애로는 Adam optimizer와 SGD를 사용해봤는데, 

1. Adam의 경우는 3개 + meta-learned된 것에 대해, 모두 잘 수행시켜주더라.
2. SGD의 경우는 meta-learned 된 것에 대해서만 잘 해주더라.

---

## #4.3 Faster Convergence

### #4.3.1 Image Regression - compare to Baselines

![스크린샷 2025-11-19 오후 2.09.06.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.09.06.png)

![위 그림에서 왼쪽에 대한 것.](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.10.51.png)

위 그림에서 왼쪽에 대한 것.

두 개를 보면 알 수 있듯이, meta-learn된 것은 human face에 조금 더 specialized돼있음.

1. 그럼에도, 오른쪽 이미지와 같이 face가 아닌 sample에 대해서도 금방 복원해내는 모습을 보여줌.
2. Meta-learn 방식에서 사용된 것은 2 gradient step만을 통해서도 PSNR이 Baseline의 거진 2배의 성능을 보여줌.

### #4.3.2 View synthesis for ShapeNet objects - compare to standard

![스크린샷 2025-11-19 오후 9.42.22.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.42.22.png)

‘의자’라는 카테고리로부터 나온 복원 정확도를 standard와 비교해서 plotting한 것.

밑이 iteration, 즉, 많은 iteration을 거친 뒤에야 그 복원률이 비슷해졌다.

---

## #4.4 Generalizing from partial observations

사실 이게 가장 중요하지 않나…

이 논문 주 목적이기도 한 것이지 않나 싶다.

### #4.4.1 Image Regression within a category

![스크린샷 2025-11-19 오후 9.46.36.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.46.36.png)

⇒ optimized된 network가 class-specific한 지를 나타내는 지표. 

- 세로는 meta-learn이 시행된 class
- 가로는 test에서 시행될 것이 속한 class를 의미.

⇒ 수치는 PSNR로 가져왔다.

⇒ 당연한 것 하나는, 자기가 meta-learn된 것에 대해서 가장 우수한 성능을 보였다는 것.

⇒ 재밌는 점 하나는, celebA, ImageNet과 같은 natural 이미지를 학습한 얘들 끼리는 나름 좋은 성능을 보였다는 것.

⇒ 근데 더 재밌는 건 SDF는 왜 높지?

⇒ 그냥 단순하게 생각하면 됨.

⇒ SDF는 그냥 단순한 선을 그려놓은 거라고 생각하면 됨.https://github.com/mintpancake/2d-sdf-net

⇒ 그냥 테스트가 너무 쉬워서 성능이 잘 나오는 거임.

⇒ 미적분 풀던 애한테 갑자기 under 9의 구구단 시험을 내는 것과 유사.

⇒ 그리고 또 재밌는 건, Text img 들에 대해서는 아무래도 다른 얘들과는 비슷하지 않으니까, text로 initialized된 것은 다른 얘들과 별로 유사하지 않으니,(frequency spectrum이 다름) 전체적으로 성능이 떨어짐.

⇒ natural img들은 색이 연속적으로 이어져있으나, text는 흰 바탕에 검은 글씨로 갑자기 바뀌니, high frequency가 생기니, 그래서 frequency spectrum이 안 맞는다는 것.

⇒ 그래서 반대로, natural img를 학습한 얘들도 text img를 학습하지 못 함.

---

### #4.4.2 CT reconstruction from sparse views

![스크린샷 2025-11-20 오전 9.34.10.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-20_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_9.34.10.png)

![스크린샷 2025-11-20 오전 9.34.22.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-20_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_9.34.22.png)

⇒ 결론적으로, Meta-learned된 것이 가장 좋은 성능을 보였고,

⇒ 중점적으로 봐야할 점은, meta-learned와 standard를 비교했을 때, standard에 사용된 view의 숫자가 2배로 늘어도, Meta-learned가 더 좋은 성능을 보였다.

(1 : 2 = # views of Meta-learned : # views of Standard)

⇒ 근데 그건 그냥 아무리 키워도 8 view라서 그런 거 아닌가

⇒ few-shot이긴 해도, 만약에 1000개 쯤 있는 상황이라 쳤을 때, 이런 양상이 계속될 수 있을까.

---

### #4.4.3 Single image view synthesis for ShapeNet

![스크린샷 2025-11-20 오전 9.54.32.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-20_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_9.54.32.png)

![스크린샷 2025-11-20 오전 9.54.13.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-20_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_9.54.13.png)

⇒ 먼저 사용된 모델은 simple-NeRF이고,

⇒ 조금 더 봐야할 점은 MV/SV Meta이다.

1. MV Meta model : meta-learn과정에서, 한 object당 25개의 view를 가지고 train을 했다는 것이다.
2. SV Meta model : meta-learn 과정에서, 한 objectekd 1의 view만 가지고 train을 했다는 것이다.

⇒ naive한 Standard Model은 형체도 알아볼 수 없게 된 데에 반해, SV Meta는 같은 조건임에도 어느정도 복원을 시키는 데에 성공.

---

### #4.4.4 View synthesis with appearance transfer for Phototourism

![스크린샷 2025-11-20 오후 5.12.45.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-20_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.12.45.png)

![스크린샷 2025-11-20 오후 5.18.51.png](Learned%20Initializations%20for%20Optimizing%20Coordinate-/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-20_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.18.51.png)

⇒ 동일 조건 내에 multi-view를 찾는 게 불가능한 상황이였음.(transient한 것들도 많고, 날씨도 다르니)

⇒ 그래서 목표는, 타겟으로하는, input view를 말하는 듯, 사진을 잘 복원하는 걸 목표로 했음.

⇒ 그래서 outer loop동안에 appearance가 잘 나오는 지를 확인함.

⇒ [#4.1.4에서 언급했듯](https://www.notion.so/Learned-Initializations-for-Optimizing-Coordinate-Based-Neural-Representations-2aaff2baaf928019b94bddd66c369d68?pvs=21), 각 landmark 당 1000개의 사진을 가져와서 학습을 진행시켰음.

문제는 PSNR 측정에서 생김.

⇒ 동일 조건의 다른 뷰를 가진 사진이 존재하지 않는다는 것이였음.

⇒ 그래서 저자들이 사용한 방식은, 사진을 절반으로 잘랐다고 생각하고, 

- 왼쪽 절반으로 최적화 과정을 가지고,
- 그 뒤에 오른쪽 절반을 그리라고 시키는 것.

⇒ Table 5에서 말하는 것이 바로 그것.

⇒ Reptile 방식으로 최적화를 진행했을 때, 기준으로 말하면,

- Standard simple-NeRF는 1 inner loop를 돌린 상태인 거고,
- meta-learned NeRF는 64번의 unrolled inner loop를 돌린 상태이며, 더 landmark에 대해, clear한 rendering이 가능해진다고 한다.

⇒ Figure 6을 간략히 설명하면, 

⇒ pose에 해당하는 최상단 row에 있는 얘들이 viewpoint에 해당한다.

⇒ 좌측 column에 있는 input view를 넘겨주면, 최적화가 끝난 모델이 예측을 진행하는 것이다.

⇒ 그 결과가 instance로 있는 것들.

---

# #5 Conclusion

결론적으로, model의 archietecture를 변경하는 것 없이, 초기 가중치를 수정하는 것만으로도 더 나은 최적화 과정을 만들어냈다.

- Next_steps
    - meta-learning algorithm을 더 정교한 걸 사용할 수도 있을 것임.
    - 또한 좋은 초기 가중치들을 설정하는 데에, sizable한, 그래도 어느정도 데이터셋 수가 받쳐줘서 좋은 초기 가중치를 뽑아냄.
    - test-time optimization도 어느 정도 필요하다.