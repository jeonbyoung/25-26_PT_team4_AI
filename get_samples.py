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



# Gemini 도움
# get_samples.py (기존 코드 아래에 이어서 붙여넣기)

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Coarse 단계에서 얻은 weights(불투명도)를 확률분포로 변환하여,
    물체가 있는 곳에 샘플(N_importance)을 집중적으로 다시 뿌리는 함수
    """
    weights = weights + 1e-5 # 0 나누기 방지
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1) 

    # 1. 0~1 사이의 난수 생성
    if det:
        u = torch.linspace(0., 1., steps=N_importance, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=bins.device)

    # 2. 역변환 샘플링 (Inverse Transform Sampling)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    
    inds_g = torch.stack([below, above], -1) 
    
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    return samples