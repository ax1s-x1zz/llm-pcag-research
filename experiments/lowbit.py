# experiments/lowbit.py
# INT8/INT6/INT5/INT4/INT3/INT2 균일 저비트 실측 양자화 엔진 (bitsandbytes 미지원 비트용).
#
# 배경:
#   bitsandbytes 는 8bit / 4bit(NF4·FP4) 만 지원한다. 따라서 FP16→INT8→INT6→INT5→INT4→
#   INT3→INT2 의 "균일 방법론" PCAG 곡선을 위해, 모든 비트폭을 본 모듈의 per-channel
#   RTN(Round-To-Nearest) 대칭 양자화 한 가지로 통일한다.
#
#   - 같은 방법을 모든 비트에 적용 → PCAG 곡선이 방법론 혼합에 오염되지 않는다.
#   - 보정(uncalibrated) 없는 단순 양자화임을 Notes 에 명시하고, 표준 방법(bitsandbytes)
#     검증 세트와는 별도 파일로 관리한다. (benchmark_driver --quant-method 참고)
#
# 특징:
#   - packed 저장: INT8=1.0B, INT6=0.75B, INT5=0.625B, INT4=0.5B, INT3=0.375B, INT2=0.25B /값
#   - 양자화 과정에서 원본 fp16 가중치는 즉시 해제되어 실질 VRAM 절감
#   - forward 시 packed → fp16 dequant → F.linear (측정용, 정확도는 근사)
#   - GPU 실측(무료 T4 16GB)에서 FP16 모델을 통째로 못 올리는 경우에도
#     packed 가중치만 GPU에 올리므로 OOM 없이 실행 가능.
#
# 호환성: Llama-3 / Qwen-2.5 / Gemma-2 / Mistral 등 nn.Linear 기반 CausalLM 모두 적용.
import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)


def _bits_to_range(bits):
    """(qmin, qmax, offset) — 대칭 양자화 범위."""
    qmax = (1 << (bits - 1)) - 1
    offset = 1 << (bits - 1)
    return -offset, qmax, offset


# ---------------------------------------------------------------- packing
def pack_lowbit(q_u, bits):
    """uint8 텐서 q_u (값 ∈ [0, 2^bits)) 를 packed uint8 로 압축.
    q_u shape: (out, in). 반환: (out, npacked). bits=8 은 그대로(1B/값)."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"pack 지원 비트: {SUPPORTED_BITS} (현재 {bits})")
    n = q_u.shape[1]
    if bits == 8:
        return q_u.to(torch.uint8).contiguous()
    if bits == 2:
        pad = (4 - (n % 4)) % 4
        if pad:
            q_u = F.pad(q_u, (0, pad))
        q4 = q_u.reshape(q_u.shape[0], -1, 4)
        b = q4[..., 0] | (q4[..., 1] << 2) | (q4[..., 2] << 4) | (q4[..., 3] << 6)
        return b.to(torch.uint8).reshape(q_u.shape[0], -1)
    if bits == 3:
        pad = (8 - (n % 8)) % 8
        if pad:
            q_u = F.pad(q_u, (0, pad))
        q8 = q_u.reshape(q_u.shape[0], -1, 8)
        v0, v1, v2, v3, v4, v5, v6, v7 = (q8[..., i] for i in range(8))
        b0 = (v0 | (v1 << 3) | (v2 << 6)).to(torch.uint8)
        b1 = ((v2 >> 2) | (v3 << 1) | (v4 << 4) | (v5 << 7)).to(torch.uint8)
        b2 = ((v5 >> 1) | (v6 << 2) | (v7 << 5)).to(torch.uint8)
        return torch.stack([b0, b1, b2], dim=-1).reshape(q_u.shape[0], -1)
    if bits == 4:
        pad = (2 - (n % 2)) % 2
        if pad:
            q_u = F.pad(q_u, (0, pad))
        q2 = q_u.reshape(q_u.shape[0], -1, 2)
        b = q2[..., 0] | (q2[..., 1] << 4)
        return b.to(torch.uint8).reshape(q_u.shape[0], -1)
    if bits == 5:
        pad = (8 - (n % 8)) % 8
        if pad:
            q_u = F.pad(q_u, (0, pad))
        q8 = q_u.reshape(q_u.shape[0], -1, 8)
        v0, v1, v2, v3, v4, v5, v6, v7 = (q8[..., i] for i in range(8))
        b0 = (v0 | ((v1 & 7) << 5)).to(torch.uint8)
        b1 = ((v1 >> 3) | ((v2 & 31) << 2) | ((v3 & 1) << 7)).to(torch.uint8)
        b2 = ((v3 >> 1) | ((v4 & 31) << 4)).to(torch.uint8)
        b3 = ((v4 >> 4) | ((v5 & 31) << 1) | ((v6 & 3) << 6)).to(torch.uint8)
        b4 = ((v6 >> 2) | ((v7 & 31) << 3)).to(torch.uint8)
        return torch.stack([b0, b1, b2, b3, b4], dim=-1).reshape(q_u.shape[0], -1)
    if bits == 6:
        pad = (4 - (n % 4)) % 4
        if pad:
            q_u = F.pad(q_u, (0, pad))
        q4 = q_u.reshape(q_u.shape[0], -1, 4)
        v0, v1, v2, v3 = (q4[..., i] for i in range(4))
        b0 = (v0 | ((v1 & 3) << 6)).to(torch.uint8)
        b1 = ((v1 >> 2) | ((v2 & 15) << 4)).to(torch.uint8)
        b2 = ((v2 >> 4) | ((v3 & 63) << 2)).to(torch.uint8)
        return torch.stack([b0, b1, b2], dim=-1).reshape(q_u.shape[0], -1)
    raise ValueError(bits)


def unpack_lowbit(packed, bits, n):
    """packed uint8 → 원래 개수 n 개의 uint8 값 (값 ∈ [0, 2^bits)). bits=8 은 그대로."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unpack 지원 비트: {SUPPORTED_BITS} (현재 {bits})")
    p = packed.to(torch.uint8)
    if bits == 8:
        return p[..., :n]
    if bits == 2:
        u = torch.empty(p.shape[0], p.shape[1] * 4, dtype=torch.uint8, device=p.device)
        u[..., 0::4] = p & 3
        u[..., 1::4] = (p >> 2) & 3
        u[..., 2::4] = (p >> 4) & 3
        u[..., 3::4] = (p >> 6) & 3
        return u[..., :n]
    if bits == 3:
        p3 = p.reshape(p.shape[0], -1, 3)
        b0, b1, b2 = p3[..., 0], p3[..., 1], p3[..., 2]
        v0 = b0 & 7
        v1 = (b0 >> 3) & 7
        v2 = ((b0 >> 6) & 3) | ((b1 & 1) << 2)
        v3 = (b1 >> 1) & 7
        v4 = (b1 >> 4) & 7
        v5 = ((b1 >> 7) & 1) | ((b2 & 3) << 1)
        v6 = (b2 >> 2) & 7
        v7 = (b2 >> 5) & 7
        u = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1).reshape(p.shape[0], -1)
        return u[..., :n]
    if bits == 4:
        u = torch.empty(p.shape[0], p.shape[1] * 2, dtype=torch.uint8, device=p.device)
        u[..., 0::2] = p & 15
        u[..., 1::2] = (p >> 4) & 15
        return u[..., :n]
    if bits == 5:
        p5 = p.reshape(p.shape[0], -1, 5)
        b0, b1, b2, b3, b4 = (p5[..., i] for i in range(5))
        v0 = b0 & 31
        v1 = (b0 >> 5) | ((b1 & 3) << 3)
        v2 = (b1 >> 2) & 31
        v3 = (b1 >> 7) | ((b2 & 15) << 1)
        v4 = (b2 >> 4) | ((b3 & 1) << 4)
        v5 = (b3 >> 1) & 31
        v6 = (b3 >> 6) | ((b4 & 7) << 2)
        v7 = (b4 >> 3) & 31
        u = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1).reshape(p.shape[0], -1)
        return u[..., :n]
    if bits == 6:
        p4 = p.reshape(p.shape[0], -1, 3)
        b0, b1, b2 = p4[..., 0], p4[..., 1], p4[..., 2]
        v0 = b0 & 63
        v1 = (b0 >> 6) | ((b1 & 15) << 2)
        v2 = (b1 >> 4) | ((b2 & 3) << 4)
        v3 = (b2 >> 2) & 63
        u = torch.stack([v0, v1, v2, v3], dim=-1).reshape(p.shape[0], -1)
        return u[..., :n]
    raise ValueError(bits)


class LowBitLinear(nn.Module):
    """packed 저비트 가중치 + 채널별 scale 로 동작하는 Linear.

    qweight: packed uint8 (실 저장), qscale: 채널별 fp16 scale.
    forward 시 dequant 후 F.linear. bias 는 그대로 유지.
    """

    def __init__(self, bits, in_features, out_features):
        super().__init__()
        self.bits = bits
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", None)
        self.register_buffer("qscale", None)
        self.register_buffer("_bias", None)
        self._orig_dtype = None

    def quantize(self, weight, bias=None):
        """fp16/bf16/fp32 Linear 가중치를 packed 저비트로 변환 (원본은 해제)."""
        dtype = weight.dtype
        self._orig_dtype = dtype
        qmin, qmax, offset = _bits_to_range(self.bits)
        wf = weight.float()
        scale = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.round(wf / scale).clamp(qmin, qmax)
        q_u = (q + offset).to(torch.uint8)
        self.qweight = pack_lowbit(q_u, self.bits)
        self.qscale = scale.half() if dtype in (torch.float16, torch.bfloat16) else scale
        self._bias = bias.detach() if bias is not None else None
        # 원본 fp16 가중치 즉시 해제 (VRAM 절감)
        del wf, q, q_u

    def forward(self, x):
        _, _, offset = _bits_to_range(self.bits)
        u = unpack_lowbit(self.qweight, self.bits, self.in_features)
        w = (u.float() - offset) * self.qscale
        w = w.to(self._orig_dtype if self._orig_dtype is not None else x.dtype)
        bias = self._bias.to(x.dtype) if self._bias is not None else None
        return F.linear(x, w, bias)


def quantize_linear(module, bits):
    """nn.Linear → LowBitLinear 변환 (in-place 아님, 새 모듈 반환)."""
    low = LowBitLinear(bits, module.in_features, module.out_features)
    low.quantize(module.weight.detach(), module.bias)
    return low


def quantize_model(model, bits):
    """모델의 모든 nn.Linear 를 LowBitLinear 로 교체 (in-place). 반환: model."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"지원 비트: {SUPPORTED_BITS} (현재 {bits})")
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            parent._modules[child] = quantize_linear(module, bits)
    return model