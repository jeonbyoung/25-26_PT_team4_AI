# Camera Parameters for Ray Casting

- Reference
    
    https://www.cs.cmu.edu/~16385/lectures/lecture10.pdf
    
    https://xoft.tistory.com/11
    
    https://xoft.tistory.com/12
    
    https://velog.io/@gjghks950/NeRF-%EA%B5%AC%ED%98%84-%ED%86%BA%EC%95%84%EB%B3%B4%EA%B8%B0-feat.-Camera-to-World-Transformation
    

NeRF 구현을 위해 ray casting을 공부한 내용을 이곳에 기록한다.

---

NeRF 구현을 위해 dataset을 다운 받으면 다음과 같이 나와있다.

```json
{
    "camera_angle_x": 0.6911112070083618,
    "frames": [
        {
            "file_path": "./train/r_0",
            "rotation": 0.012566370614359171,
            "transform_matrix": [
                [
                    -0.9999021887779236,
                    0.004192245192825794,
                    -0.013345719315111637,
                    -0.05379832163453102
                ],
                [
                    -0.013988681137561798,
                    -0.2996590733528137,
                    0.95394366979599,
                    3.845470428466797
                ],
                [
                    -4.656612873077393e-10,
                    0.9540371894836426,
                    0.29968830943107605,
                    1.2080823183059692
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        {
            "file_path": "./train/r_1",
            "rotation": 0.012566370614359171,
            "transform_matrix": [
                [
                    -0.9305422306060791,
                    0.11707554012537003,
                    -0.34696459770202637,
                    -1.398659110069275
                ],
                [
                    -0.3661845624446869,
                    -0.29751041531562805,
                    0.8817007541656494,
                    3.5542497634887695
                ],
                [
                    7.450580596923828e-09,
                    0.9475130438804626,
                    0.3197172284126282,
                    1.2888214588165283
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
```

~~이게 뭘까~~

일단 자료를 조금 찾아보다 보면 결국 이런 형식으로 나옴을 알 수 있다.

```json
{
    "camera_angle_x": 0.6911112070083618,
    "frames": [
        {
            "file_path": "./train/r_0",
            "rotation": 0.012566370614359171,
            "transform_matrix": [
                [
                    //r11,
                    //r12,
                    //r13,
                    //tx
                ],
                [
                    //r21,
                    //r22,
                    //r23,
                    //ty
                ],
                [
                    //r31,
                    //r32,
                    //r33,
                    //tz
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
```

즉, $\begin{bmatrix} r_{11} &r_{12} &r_{13} &t_x \\ r_{21} &r_{22} &r_{23}&t_y \\r_{31} &r_{32} &r_{33} &t_z \\ 0&0&0&1 \end{bmatrix}$를 의미한다. 위에 태그에서 볼 수 있듯이, 이는 transfom의 기능을 해주는데, 정확히 **무엇을 무엇으로** transform하는 것인지 확인해보자.

우선 좌표계를 헷갈리면 안 된다.

![image.png](Camera%20Parameters%20for%20Ray%20Casting/image.png)

위와 같이, 바라보는 시점을 기준으로 x-y를 나누며, 사실 생각해보면 당연하다. 결국엔 2D로 변환할 건데 x-y 평면이 내가 바라보는 시점과 수직해야할테니까.

그래서 뒤쪽에 Z의 값에 -1을 곱하라는 의미도 이런 의미에서 비롯된다. 우리는 앞을 보고 있는데 Z축의 양의 방향은 우리를 향해 있으니.

카네기 멜론의 slide에서 볼 수 있듯이, 결국 coordinate transform이란 아래와 같다.

- coordinate transformation이란, 결국 $x_c=PX_w$를 해결하는 것.
    - $x_c~~(3\times1)$ : 2D img point
    - $P~~(3\times4)$ : camera matrix
    - $X_w~~(4\times1)$ : 3D world point
    
    ⇒ 결국 세상의 값(world coordinate)을 img coordinate으로 환산하는 것을 말한다.
    
    ⇒ 그리고, 이것이 img→world로 바로 변환되는 관계인가?
    
    ⇒ **그렇지 않다.** img로 표현된다는 것은 중간에 **카메라** 혹은 그 역할을 해주는 것이 개입되기 마련이다.
    
    ⇒ 즉, 봐야할 것은 변환 행렬인 camera matrix이다.
    

---

# #1 Camera Matrix consideration

**CMU의 slide를 많이 참고했는데, 여기선 pinhole camera를 가지고 얘기를 한다.**

일단 중요한 게 뭘까? 이 matrix는 **어떤 특징들**을 담고 있어야할까?

이에 대해 2가지 특징을 주로 삼았다.

1. focal length
    
    ![스크린샷 2025-11-26 오후 1.41.55.png](Camera%20Parameters%20for%20Ray%20Casting/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-26_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.41.55.png)
    
    ⇒ 위와 같이, img의 사이즈에 영향을 주게 된다.
    
2. Pinhole size
    
    ![스크린샷 2025-11-26 오후 1.42.50.png](Camera%20Parameters%20for%20Ray%20Casting/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-26_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.42.50.png)
    
    위와 같이, pinhole의 크기가 커졌을 때, 더 빛이 퍼지며 들어와, 더 blurry해지게 된다.
    

이 외에도 짧게 렌즈에 대해서도 알아봤는데, 여기서 렌즈를 중간에 낄 때는 focal length가 pinhole일 때와는 다르게 작용한다고 한다.

![스크린샷 2025-11-26 오후 1.44.30.png](Camera%20Parameters%20for%20Ray%20Casting/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-26_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.44.30.png)

⇒ 근데 pinhole만 보기로 했는데, 왜 갑자기 lens가 나오나?

![스크린샷 2025-11-26 오후 1.46.00.png](Camera%20Parameters%20for%20Ray%20Casting/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-26_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.46.00.png)

⇒ 이 그림에서 알 수 있듯이, pinhole이나, lens나 central ray가 움직이는 경로는 동일하다!

⇒ 즉, 일반적으로 pinhole을 통해 사진을 찍지 않고 lens를 사용하지만, 지금 하는 공부가 틀어지지는 않는다는 내용이다.

---

## #1.1 2D Scaling (Focal Length)

일반적으로 사진을 찍는 것을 생각해보자. 그리고 이에 대해, 연산의 편의를 위해, 상이 맺히는, img plane에 놓여지는 부분을 real-world object 와 겹쳐 올릴 것이다. 아래와 같이 말이다.

![image.png](Camera%20Parameters%20for%20Ray%20Casting/image%201.png)

![image.png](Camera%20Parameters%20for%20Ray%20Casting/image%202.png)

이것을 이제 데카르트 좌표계 위에 단순화 해서 올려보자

![스크린샷 2025-11-26 오후 1.56.33.png](Camera%20Parameters%20for%20Ray%20Casting/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-26_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.56.33.png)

그렇다면, 이런 모양이 되고, z=1인 이유는, 의도적으로 그렇게 잡은 것이다. focal length = 1로 설정한 상태로 구하는 것이다(나중에 일반화 시킬 것이다.)

- focal length 구하는 방법
    
    ![image.png](Camera%20Parameters%20for%20Ray%20Casting/image%203.png)
    
    위 그림에서 img plane에 들어오는 값을 나타낸 것이다.
    
    - $f$ : focal length를 나타낸다.
    - $\theta$ : pinhole로 들어오는 맨 끝 ray들 간의 각도
    
    $f$로 나눠진 삼각형 중 위 삼각형만 봤을 때,
    
    ⇒ $f = \frac{1}{2}W \frac{1}{\tan{ \frac{\theta}{2} }}$로 나타낼 수 있게 된다.
    

이 상황에서 y-z 평면(x축 방향에서)을, 그리고 x-z 평면(y축 방향에서)을 바라본 모습으로 바꿔보자.

![image.png](Camera%20Parameters%20for%20Ray%20Casting/image%204.png)

![image.png](Camera%20Parameters%20for%20Ray%20Casting/image%205.png)

즉, 우리는 이 식을 통해, img plane에서 x,y에 해당하는 값들이 어떻게 구성돼있는 지를 알 수 있게 된다.

그럼 이제 일반화시켜보자. focal length가 항상 1일리는 없지 않는가.

위 비례식에서 그냥 1대신에 $f$를 넣었다고 생각하면, 식은 아래와 같이 구성된다.

$\begin{bmatrix} X \\ Y \\ Z\end{bmatrix} ->\begin{bmatrix} \frac{fX}{Z} \\ \frac{fY}{Z} \end{bmatrix}$

여기서 화살표 부분을 만들어내며 생각해보자. 다시 상기해볼 점은 우린, $x=PX$ , 특히 $P$를 구하려고 하는 것이다.

그리고 $\begin{bmatrix} X \\ Y \\ Z\end{bmatrix}$는 world coordinate을 의미하고, $\begin{bmatrix} \frac{fX}{Z} \\ \frac{fY}{Z} \end{bmatrix}$는 img coordinate을 의미한다. 

즉, 결과적으로 얻고자 하는 건 $x = \begin{bmatrix} \frac{fX}{Z} \\ \frac{fY}{Z} \end{bmatrix}$, $X = \begin{bmatrix} X \\ Y \\ Z\end{bmatrix}$으로의 변환을 도와주는 $P$이다.

그래서 $P=\begin{bmatrix} f& 0& 0& 0 \\ 0 &f & 0& 0 \\ 0& 0& 1& 0 \end{bmatrix}$으로 구성된다고 한다.

이것을 이후의 기능들을 추가했을 때, 직관적으로 잘 보이게 하기 위해, $P$를  $(3\times3)\times(3\times4)$ 으로 맞추기로 했다. 그래서, $\begin{bmatrix} I~|~0 \end{bmatrix}$짜리를 곱한 형태를 다음과 같이 맞춰준다.

$$
P=\begin{bmatrix} f& 0& 0 \\ 0 &f & 0 \\ 0& 0& 1&\end{bmatrix}\begin{bmatrix} 1& 0& 0&|~0 \\ 0 &1 & 0& |~0 \\ 0& 0& 1&|~0 \end{bmatrix}
$$

근데 이상한 점이 있다. 왜 $f$를 곱해주는 건 알겠는데, 왜 $Z$로 나눠주는 건 없을까.

그리고 $Z$는 왜 1을 그냥 곱할까. 우리가 img plane으로 바꾸면서 원하는 것은 $Z$축을 눌러서, 2D로 바꾸는 것인데.

찾아본 바로는 이 전체에 $Z$를 나누는 것이 필요한 것 같다. 그럼 큰 문제가 없어보인다. 근데 이 부분이 생략됐다고 한다. 이유는 조금 더 찾아보는 걸로…

---

## #1.2 Translation or Shifting (camera origin → img origin)

#1.1에서 우리가 해결한 부분은 focal length, 즉, 카메라와 img가 상이 맺히는 plane과의 거리였다.

이 다음은 **원점(origin)**을 맞추는 단계이다.

![스크린샷 2025-11-26 오후 2.33.00.png](Camera%20Parameters%20for%20Ray%20Casting/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-26_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.33.00.png)

일반적으로 camera와 img는 서로 다른 coordinate system을 갖고 있다. 이를 맞춰주는 것이다.

이는 다소 단순히 진행된다.

![image.png](Camera%20Parameters%20for%20Ray%20Casting/image%206.png)

$p$만큼 떨어져있을 때, 그에 대해, $x,y$ 각각으로 해당 거리만큼 간다는 것이다.

더 정확히 하면, 사실 원점은 저 위치가 아닐 수 있다.

1. origin이 좌하단인 경우 : 주로 수학, 더 세분화 되면 OpenGL에 쓰인다고 한다.
2. origin이 좌상단인 경우 : 주로 컴퓨터, OpenCV, Numpy 분야.

**⇒ 즉, 카메라는 중앙에서 시작하는데, pinhole만 봐도 그렇지 않은가, img는 origin이 다르다는 점에서 이 짓을 하는 것이다.**

이것까지 반영하게 되면, $P$는 다음과 같이 구성된다.

$$
P=\begin{bmatrix} f& 0& p_x \\ 0 &f & p_y \\ 0& 0& 1&\end{bmatrix}\begin{bmatrix} 1& 0& 0&|~0 \\ 0 &1 & 0& |~0 \\ 0& 0& 1&|~0 \end{bmatrix}
$$

왜 저곳인가?

⇒ 결과적으로 봤을 때, 오른쪽에는 $\begin{bmatrix} X \\ Y \\ Z \\1\end{bmatrix}$이 곱해진다. 그렇다면, 

결국은 $Z$에 대해, shifting multiplier들이 곱해지게 되는데, 결국 형태는 이렇게 된다.

$$
x = fX+Zp_x \\y=fY + Zp_y
$$

결국, $Z$가 곱해질 것도 생각하면 다음과 같이 정리된다.

$$
x' = fX/Z+p_x \\y'=fY/Z + p_y
$$

즉, $x,y$에 대해 옮겨진 형태이다.

근데 이것도 위에서 한 것처럼 조금 나눠서 쓰게 되면, 

$$
P=\begin{bmatrix} 1& 0& p_x \\ 0 &1 & p_y \\ 0& 0& 1&\end{bmatrix}\begin{bmatrix} f& 0& 0 \\ 0 &f & 0 \\ 0& 0& 1&\end{bmatrix}\begin{bmatrix} 1& 0& 0&|~0 \\ 0 &1 & 0& |~0 \\ 0& 0& 1&|~0 \end{bmatrix}
$$

---

## #1.3 Shearing

⇒ 기울어진 정도를 의미한다.

![ref) [https://xoft.tistory.com/12](https://xoft.tistory.com/12)](Camera%20Parameters%20for%20Ray%20Casting/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-26_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.01.35.png)

ref) [https://xoft.tistory.com/12](https://xoft.tistory.com/12)

위 그림과 표현이 가능한데, 사실 NeRF에서 이에 대한 구현을 진행하진 않았다.

~~그래서 급한 불부터 끄고 다음에 보자.~~

---

## #1.4 Rotation

지금까지 한 것을 정리해보면,

1. focal length를 통해서, scaling을 해줬고
2. origin을 맞춤으로써, camera - img를 맞춰줬다.

이번에 할 것은 camera - world 를 맞춰주는 것이다.

더 정확히는 두 개의 origin을 맞춰주고, world coordinate에 있는 것이 정확히 camera coordinate에 있는 애에 mapping되게 해주는 것이다. 아래 그림을 보자.

![스크린샷 2025-11-26 오후 6.08.09.png](Camera%20Parameters%20for%20Ray%20Casting/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-26_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.08.09.png)

- $\~{C}$ : camera coordinate의 중심
- $\tilde{X_w}$ : world coordinate에 있는 한 점.

⇒ 우리가 이제 할 거는 $\tilde{X_w}$ 가, camera coordinate에 안전하게 올라가는 것을 하면 된다.

그걸 위해서 다음을 수행한다.

$$
\tilde{X_w}-\tilde{C}
$$

직관적으로 생각하면 더 편하다. $\tilde{X_w}$가 뭔지는 모르겠으나, 확실한 건 camera coordinate 밖에 있는, 근데 차원 수는 같은 애다.

그러니, camera coordinate으로 끌고 오려면, 그 중심값을 빼줌으로써 가져온다는 느낌이다. 약간 아래 평행이동 그림이랑도 같다고 생각한다. ~~개인적으로는~~

![ref) [https://suhak.tistory.com/1495](https://suhak.tistory.com/1495)](Camera%20Parameters%20for%20Ray%20Casting/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-26_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.15.16.png)

ref) [https://suhak.tistory.com/1495](https://suhak.tistory.com/1495)

다시 돌아와서, 일단 그렇게 맞춰줬다.

근데 문제가 생긴 게, allign(합쳐진다) 되지가 않을 가능성이 높다는 것이다!

왤까?

![스크린샷 2025-11-26 오후 6.25.39.png](Camera%20Parameters%20for%20Ray%20Casting/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-26_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.25.39.png)

위 그림에서처럼 두 system의 회전 정도가 일치하지 않아서라는데, 난 이렇게 해석했다.

![image.png](Camera%20Parameters%20for%20Ray%20Casting/image%207.png)

**기준은 world다!**

camera의 **Z축만 y의 양의 방향으로, 그리고 x축의 음의 방향으로** 당긴 것이라고 하자.

두 system을 합쳐보면, 어느 정도 회전의 오차가 생기게 되고, 그에 따라 point들이 그대로 allign되지 않을 수 있다.

이런 world와의 회전 정도로 인한 오차!

⇒ 그걸 맞춰주는 것이 이제 구할 $R$이라는 matrix다. 다음과 같이 구성할 것이다.

$$
R=\begin{bmatrix} r_{xx} &r_{xy} &r_{xz} \\ r_{yx} &r_{yy} &r_{yz}\\r_{zx} &r_{zy} &r_{zz} \end{bmatrix}
$$

위와 그림과 같은 상황이라고 생각하고, 간단히 $r_{xz}, r_{yz}$만 따서 와보자.

위와 그림과 같은 상황이라고 생각하고, 간단히 $r_{xz}, r_{yz}$만 따서 와보자.

- $r_{xz}<0$ : $z$축이, $x$축에 대해, 음의 방향으로 돌아가있다.
- $r_{yz}>0$ : $z$축이, $y$축에 대해, 양의 방향으로 돌아가있다.

⇒ 또한 각 element는 ≤ 1의 상태를 가진다. 결국, 이는 돌아간 비율 관계이니.

이런 rotation까지 적용시킨 것이 최종 형태가 될 것이다. 따라서 수식은 다음과 같아진다.

$$
\tilde{X_c} = R \cdot(\tilde{X_c}-\tilde{C})
$$

이를 y,z 까지 확장하고, vector form으로 나타낸다고 치면,

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \\1\end{bmatrix} = \begin{bmatrix} R & -R\~{C} \\ 0 & 1 \end{bmatrix}\begin{bmatrix}X_w\\Y_w\\ Z_w \\ 1 \end{bmatrix}
$$

위와 같이 된다.

$R$은  $3\times3$, $\~{C}=\begin{bmatrix} t_x\\t_y\\t_z \end{bmatrix}$ 는 $3\times1$이니, 결국, $\begin{bmatrix} R & -R\~{C} \\ 0 & 1 \end{bmatrix} = 3\times4 ~matrix$가 된다. 

($t_x,t_y,t_z$는 camera coordinate의 origin을 의미한다.)

---

# #2 Incorporating considerations

결국, 그동안 봐온 #1.1,2,4의 내용을 합치게 되면,

$P$를 다음과 같이 구성할 수 있다.

$$
P=\begin{bmatrix} 1& 0& p_x \\ 0 &1 & p_y \\ 0& 0& 1&\end{bmatrix}\begin{bmatrix} f& 0& 0 \\ 0 &f & 0 \\ 0& 0& 1&\end{bmatrix}\begin{bmatrix} 1& 0& 0&|~0 \\ 0 &1 & 0& |~0 \\ 0& 0& 1&|~0 \end{bmatrix} \begin{bmatrix} R & -RC \\ 0 & 1 \end{bmatrix}
$$

여기서 $K= \begin{bmatrix} 1& 0& p_x \\ 0 &1 & p_y \\ 0& 0& 1&\end{bmatrix}\begin{bmatrix} f& 0& 0 \\ 0 &f & 0 \\ 0& 0& 1&\end{bmatrix}$라고 간소화시키면서 형식을 나누는데,

cmu에서 표기하는 방식은 두 가지다. 각각 의미가 다르다.

1. translate first then rotate
    
    $P=KR[I|-C]$
    
    $\because$ $[I|-C]$ ← 이것이 결국은 camera로 translation하는 것이기에
    
2. rotate first then translate
    
    $P=K[R|t] ~~~s.t. ~~t=-RC$ 
    

2번의 방식이 주로 쓰이며, 이것을 가지고, intrinsic/extrinsic을 구분하기도 한다.

- $K$가 intrinsic matrix, 카메라 안에서 일어나는, focal length, shift  등에 대한 성질을 담고,
- $[R|t]$가 extrinsic matrix, 카메라 밖, world와의 관계를 나타내는 값이다.

---

# #3 How to apply to NeRF?

결국 해낸 것은 무엇인가?

world coordinate → img coordinate으로 바꾸는 것을 한 것이다.

근데 우리가 갖고 있는 건 사실 img coordinate이지 않는가

그래서 그냥 단순하게 world coordinate인 $X_w$가 궁금하다면,

기존 식인 $x = PX_w$를, $X_w = P^{-1}x$로 구하면 된다.라고 단순히 생각했지만,

$P$가 애초에 무엇인가?

⇒ 3D → 2D로 변환해주는 애라, 이것의 역함수에서는 depth(Z축에 대한 것) 정보가 사라지게 된다.

그래서 카메라 외부 즉, extrinsic matrix의 inverse만 구하고, 그 후속과정을 해주면, $X_w$를 할 수 있을 것이다!

**⇒ 그리고 json 파일 내에 있는 transform matrix가 바로 $[R|t]^{-1}$이다.**

⇒ 위에서 언급했듯, 이 값의 구성은 $\begin{bmatrix} r_{11} &r_{12} &r_{13} &t_x \\ r_{21} &r_{22} &r_{23}&t_y \\r_{31} &r_{32} &r_{33} &t_z \\ 0&0&0&1 \end{bmatrix}$이다.

⇒ 즉, 위에서 rotation 행렬을 구성했을 때처럼, 

- 좌상단 $3\times3$행렬은 한 축에 대해 돌아간 정도를 의미하고,
- 우측 $t$ 로 구성된  $3\times1$행렬은 위치에 대한 값이 들어간다.

그래서 우리가 할 것은 $[R|t]^{-1}x=KX_w$까지 온 상황이니, 저 K만 해결해주면 된다.

그대로 K의 역을 좌변 결과에 수행해주면 된다.

$K= \begin{bmatrix} 1& 0& p_x \\ 0 &1 & p_y \\ 0& 0& 1&\end{bmatrix}\begin{bmatrix} f& 0& 0 \\ 0 &f & 0 \\ 0& 0& 1&\end{bmatrix}$라는 점에서 알 수 있듯, 우리가 돌려놔야하는 건 

1. focal length를 통한 scaling
2. shifting이다.

즉,

1. $x = x/f$로 
2.  $x = x-W/2$로 world coordinate의 origin인 중앙으로 보낸다.(기존에는 $p_x$등을 더하는 형태였으니, 이번엔 빼는 것이다.)

⇒ $x' = (x-\frac{W}{2})/f$

근데?
⇒ 내가 사용하는 OpenCV에서는 좌상단에 맞춰서 인식을 시작한다.

그에 따라 가져올 때, 

![image.png](Camera%20Parameters%20for%20Ray%20Casting/image%208.png)

다음과 같이 동작하므로, $Y_w$를 가져오게 될 때는 -를 붙여서 가져오게 된다.

즉, $y' = (y-\frac{W}{2})/f$

이 과정을 거쳐서 나오는 것이 무엇이냐?

우리의 시작으로 되돌아가면 알 수 있다.

![스크린샷 2025-11-26 오후 7.48.33.png](Camera%20Parameters%20for%20Ray%20Casting/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-11-26_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_7.48.33.png)

바로 광선에 대한 정보들이다.

1. 어디서 시작해서 ($o ~~from~~t$ )
2. 어느 방향으로 나가는 지 ($d~~from ~~R$)

에 대한 정보를 알 수 있게 된다.

이렇게 ray casting을 할 수 있다.

이후, coarse-to-fine 방식을 사용할 수도 있지만, 시간관계상, coarse만 사용해서, 임의 sampling으로 world coordinate을 뽑아낼 수 있다.