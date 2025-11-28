import torch

def get_samples(ray_direction, ray_origin, near=2.0, far=6.0, num_of_samples = 64, mode = 'train'):
    # 아래에서 .to(ray_origin.device)를 하는 이유는, ray_origin같은 tensor는 이미 GPU에 올려진 상태이다.
    # 근데, linspace로 형성된 저 samples라는 애는 CPU에 올라가있다.
    # PyTorch는 다른 곳에 있는 얘들을 서로 연산할 수 X => 그래서 같은 위치에 있게 하는 거다.

    device = ray_origin.device

    num_of_rays = ray_origin.shape[0]
    step_size = (far-near)/num_of_samples

    # 이제 광선에서 샘플을 추출하는 식은 r(t) = o + t*d를 진행.
    t_values = torch.linspace(near, far, num_of_samples).to(ray_origin.device)
    t_values = t_values.expand(num_of_rays, num_of_samples)

    if mode == 'train':
        # train에서는 statrtified 방식을 적용하여, 각 point별 간격이 달라지게 했다.
        noise = torch.rand_like(t_values)
        t_values = t_values.clone() + noise

    elif mode == 'test':
        # test에서는 그냥 중간에 가서 있게 한다.
        t_values = t_values+0.5

    t_values = near + t_values*step_size

    # 결국 sample은 다음과 같이 구성됨. [ray의 개수 | 각 ray의 sample | 각 sample은 x y z 의 좌표를 가짐.]
    samples = ray_origin[..., None, :] + ray_direction[..., None, :]*t_values[..., None]

    return samples, t_values