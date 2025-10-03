# quant_nbit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ---------- core helpers (uniform quant) ----------
def qparams_from_minmax(xmin, xmax, n_bits=8, unsigned=False, eps=1e-12):
    """
    Returns (scale, zero_point, qmin, qmax) for uniform quant.
    - unsigned=True  -> [0, 2^b - 1]
    - unsigned=False -> symmetric int range [-2^(b-1)+1, 2^(b-1)-1]
    """
    if unsigned:
        qmin, qmax = 0, (1 << n_bits) - 1
        # (common for post-ReLU) ensure non-negative min for tighter range
        xmin = torch.zeros_like(xmin)
        scale = (xmax - xmin).clamp_min(eps) / float(qmax - qmin)
        zp = torch.round(-xmin / scale).clamp(qmin, qmax)
    else:
        qmax = (1 << (n_bits - 1)) - 1
        qmin = -qmax
        max_abs = torch.max(xmin.abs(), xmax.abs()).clamp_min(eps)
        scale = max_abs / float(qmax)
        zp = torch.zeros_like(scale)
    return scale, zp, int(qmin), int(qmax)

def quantize(x, scale, zp, qmin, qmax):
    q = torch.round(x / scale + zp)
    return q.clamp(qmin, qmax)

def dequantize(q, scale, zp):
    return (q - zp) * scale

# ---------- activation fake-quant (with calibration then freeze) ----------
class ActFakeQuant(nn.Module):
    """
    Per-tensor activation fake-quant with configurable bits.
    Intended to be placed AFTER ReLU -> use unsigned=True.
    """
    def __init__(self, n_bits=8, unsigned=True):
        super().__init__()
        self.n_bits = n_bits
        self.unsigned = unsigned
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin, self.qmax = None, None

    @torch.no_grad()
    def observe(self, x):
        self.min_val = torch.minimum(self.min_val, x.min())
        self.max_val = torch.maximum(self.max_val, x.max())

    @torch.no_grad()
    def freeze(self):
        scale, zp, qmin, qmax = qparams_from_minmax(
            self.min_val, self.max_val, n_bits=self.n_bits, unsigned=self.unsigned
        )
        self.scale.copy_(scale)
        self.zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            self.observe(x)
            return x
        q = quantize(x, self.scale, self.zp, self.qmin, self.qmax)
        return dequantize(q, self.scale, self.zp)

# ---------- weight fake-quant wrappers (freeze-from-weights) ----------
class QuantConv2d(nn.Conv2d):
    """
    Per-tensor symmetric int quantization for weights with configurable bits.
    We compute/freeze params once from trained weights (PTQ).
    """
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        w = self.weight.detach().cpu()
        w_min, w_max = w.min(), w.max()
        scale, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.w_scale.copy_(scale)
        self.w_zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        q = quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)
        w_dq = dequantize(q, self.w_scale, self.w_zp)
        return F.conv2d(x, w_dq, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QuantLinear(nn.Linear):
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        w = self.weight.detach().cpu()
        w_min, w_max = w.min(), w.max()
        scale, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.w_scale.copy_(scale)
        self.w_zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.linear(x, self.weight, self.bias)
        q = quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)
        w_dq = dequantize(q, self.w_scale, self.w_zp)
        return F.linear(x, w_dq, self.bias)

# ---------- model surgery with user-selected bits ----------
def swap_to_quant_modules(model, weight_bits=8, act_bits=8, activations_unsigned=True):
    """
    - Replace every Conv2d/Linear with Quant* using weight_bits.
    - Replace every ReLU with Sequential(ReLU, ActFakeQuant(act_bits)).
    """
    for name, m in list(model.named_children()):
        swap_to_quant_modules(m, weight_bits, act_bits, activations_unsigned)

        if isinstance(m, nn.Conv2d):
            q = QuantConv2d(
                m.in_channels, m.out_channels, m.kernel_size,
                stride=m.stride, padding=m.padding, dilation=m.dilation,
                groups=m.groups, bias=(m.bias is not None),
                weight_bits=weight_bits
            )
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.Linear):
            q = QuantLinear(m.in_features, m.out_features, bias=(m.bias is not None), weight_bits=weight_bits)
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.ReLU):
            seq = nn.Sequential(OrderedDict([
                ("relu", nn.ReLU(inplace=getattr(m, "inplace", False))),
                ("aq", ActFakeQuant(n_bits=act_bits, unsigned=activations_unsigned)),
            ]))
            setattr(model, name, seq)

def freeze_all_quant(model):
    """
    Freeze weights and activations (finalize scales/ZPs) after calibration.
    """
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.freeze()
        if isinstance(mod, nn.Sequential):
            for sub in mod.modules():
                if isinstance(sub, ActFakeQuant):
                    sub.freeze()

def model_size_bytes_fp32(model):
    """Total size of all parameters if stored as FP32 (4 bytes each)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * 4
    return total

def model_size_bytes_quant(model, weight_bits=8):
    """Total size if all weights were stored as intN, biases stay FP32."""
    total = 0
    for name, p in model.named_parameters():
        if "weight" in name:
            total += p.numel() * weight_bits // 8  # intN
        elif "bias" in name:
            total += p.numel() * 4                 # keep biases FP32
    return total

def print_compression(model, weight_bits=8, act_bits=8, device="cpu"):
    """
    Prints full compression summary:
    (a) Model compression ratio (weights + biases)
    (b) Weight-only compression ratio
    (c) Activation compression ratio (measured via dummy forward pass)
    (d) Final quantized model size (MB)
    """
    # --- sizes in bytes ---
    fp32_size = model_size_bytes_fp32(model)           # all params as FP32
    quant_size = model_size_bytes_quant(model, weight_bits)  # weights intN, biases FP32

    # (a) Model compression ratio
    model_ratio = fp32_size / max(quant_size, 1)

    # (b) Weight-only compression ratio
    total_weights = sum(p.numel() for n, p in model.named_parameters() if "weight" in n)
    fp32_weight_bits = total_weights * 32
    quant_weight_bits = total_weights * weight_bits
    weight_ratio = fp32_weight_bits / quant_weight_bits

    # (c) Activation compression ratio
    activations = []

    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            activations.append(out.numel())

    handles = []
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6, ActFakeQuant)):
            handles.append(m.register_forward_hook(hook_fn))

    dummy = torch.randn(1, 3, 32, 32).to(device)
    model.eval()
    with torch.no_grad():
        model(dummy)

    for h in handles:
        h.remove()

    total_acts = sum(activations)
    fp32_act_bits = total_acts * 32
    quant_act_bits = total_acts * act_bits
    act_ratio = fp32_act_bits / quant_act_bits

    # (d) Final quantized model size in MB
    final_size_mb = quant_size / (1024 * 1024)

    # --- print summary ---
    print("=== Compression Summary ===")
    print(f"FP32 model size:   {fp32_size/1024/1024:.2f} MB")
    print(f"Quantized size:    {final_size_mb:.2f} MB (weights={weight_bits}-bit, activations={act_bits}-bit)")
    print(f"(a) Model compression ratio (weights+biases): {model_ratio:.2f}x")
    print(f"(b) Weight-only compression ratio: {weight_ratio:.2f}x")
    print(f"(c) Activation compression ratio: {act_ratio:.2f}x "
          f"(measured with CIFAR-10 dummy input)")
    print(f"(d) Final quantized model size (MB): {final_size_mb:.2f}x ")
    print("===========================")
