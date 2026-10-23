"""allin1이 쓰는 NATTEN 0.17 함수 4개를 순수 PyTorch로 다시 구현한 것.

NATTEN 0.17은 Python 3.12·CUDA용 휠만 있어서 Colab(Python 3.13)과 맥에서는 설치되지 않는다.
allin1 1.1.0은 `natten.functional`의 natten1dqkrpb / natten1dav / natten2dqkrpb / natten2dav만 쓰므로,
`use()`로 이 모듈을 `natten`으로 등록하면 CPU·CUDA·MPS 어디서든 allin1을 돌릴 수 있다.

이웃 정의는 NATTEN과 같다. dilation d이면 위치를 d로 나눈 나머지별로 묶고, 각 묶음 안에서
자기 위치를 가운데에 두는 kernel_size개 창을 잡되 끝에서는 창을 안쪽으로 민다.
상대 위치 편향(rpb) 인덱스는 (이웃 위치 - 자기 위치) / d + kernel_size - 1.
"""
import sys

import torch

__version__ = '0.17-torch'


def _neighbors(length, kernel_size, dilation, device):
    """길이 length인 축에서 위치마다 이웃 위치 (length, k)와 rpb 인덱스 (length, k)."""
    i = torch.arange(length, device=device)
    r, g = i % dilation, i // dilation
    group_len = (length - r + dilation - 1) // dilation
    if int(group_len.min()) < kernel_size:
        raise ValueError('축 길이 %d가 kernel_size %d × dilation %d보다 짧습니다' % (length, kernel_size, dilation))
    start = torch.minimum((g - kernel_size // 2).clamp(min=0), group_len - kernel_size)
    k = torch.arange(kernel_size, device=device)
    idx = (start[:, None] + k) * dilation + r[:, None]
    pb = start[:, None] + k - g[:, None] + kernel_size - 1
    return idx, pb


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    """query, key: (B, heads, L, D) → 점수 (B, heads, L, k)."""
    idx, pb = _neighbors(query.shape[2], kernel_size, dilation, query.device)
    attn = torch.einsum('bhld,bhlkd->bhlk', query, key[:, :, idx])
    return attn + rpb[:, pb] if rpb is not None else attn


def natten1dav(attn, value, kernel_size, dilation):
    """attn: (B, heads, L, k), value: (B, heads, L, D) → (B, heads, L, D)."""
    idx, _ = _neighbors(value.shape[2], kernel_size, dilation, value.device)
    return torch.einsum('bhlk,bhlkd->bhld', attn, value[:, :, idx])


def _neighbors2d(shape, kernel_size, dilation, device):
    (ih, ph), (iw, pw) = (_neighbors(n, kernel_size, dilation, device) for n in shape)
    return (ih[:, None, :, None], iw[None, :, None, :]), (ph[:, None, :, None], pw[None, :, None, :])


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    """query, key: (B, heads, H, W, D) → 점수 (B, heads, H, W, k*k)."""
    B, h, H, W, _ = query.shape
    (ih, iw), (ph, pw) = _neighbors2d((H, W), kernel_size, dilation, query.device)
    attn = torch.einsum('bhxyd,bhxyijd->bhxyij', query, key[:, :, ih, iw])
    if rpb is not None:
        attn = attn + rpb[:, ph, pw]
    return attn.reshape(B, h, H, W, kernel_size * kernel_size)


def natten2dav(attn, value, kernel_size, dilation):
    """attn: (B, heads, H, W, k*k), value: (B, heads, H, W, D) → (B, heads, H, W, D)."""
    B, h, H, W, _ = value.shape
    (ih, iw), _ = _neighbors2d((H, W), kernel_size, dilation, value.device)
    attn = attn.reshape(B, h, H, W, kernel_size, kernel_size)
    return torch.einsum('bhxyij,bhxyijd->bhxyd', attn, value[:, :, ih, iw])


def use():
    """이 모듈을 `natten`, `natten.functional`로 등록한다. allin1을 import하기 전에 부른다."""
    module = sys.modules[__name__]
    sys.modules['natten'] = module
    sys.modules['natten.functional'] = module
    module.functional = module
