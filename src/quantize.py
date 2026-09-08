# src/quantize.py
# Custom quantization implementation (no PyTorch quantization API).
# Supports configurable bit-widths for weights and activations,
# per-layer quantization control, PTQ calibration, QAT with STE,
# and actual intN weight storage with bit-packing.

import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ============================================================
# Core quantization helpers (uniform quantization)
# ============================================================

def qparams_from_minmax(xmin, xmax, n_bits=8, unsigned=False, eps=1e-12):
    """
    Compute (scale, zero_point, qmin, qmax) for uniform quantization.
    - unsigned=True  -> range [0, 2^b - 1]  (post-ReLU activations)
    - unsigned=False -> symmetric [-2^(b-1)+1, 2^(b-1)-1]  (weights)
    """
    if unsigned:
        qmin, qmax = 0, (1 << n_bits) - 1
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
    """Quantize tensor x: round(x/scale + zp) clamped to [qmin, qmax]."""
    return torch.clamp(torch.round(x / scale + zp), qmin, qmax)


def dequantize(q, scale, zp):
    """Dequantize: (q - zp) * scale."""
    return (q - zp) * scale


def fake_quantize(x, scale, zp, qmin, qmax):
    """Fake quantization: quantize then dequantize (stays in float)."""
    q = quantize(x, scale, zp, qmin, qmax)
    return dequantize(q, scale, zp)


class StraightThroughEstimator(torch.autograd.Function):
    """STE: forward does fake-quant, backward passes gradients through."""
    @staticmethod
    def forward(ctx, x, scale, zp, qmin, qmax):
        return fake_quantize(x, scale, zp, qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def ste_fake_quantize(x, scale, zp, qmin, qmax):
    """Fake quantize with straight-through estimator for QAT."""
    return StraightThroughEstimator.apply(x, scale, zp, qmin, qmax)


# ============================================================
# Bit-packing for actual low-bit storage
# ============================================================

def _int_dtype_for_bits(n_bits):
    """Return the smallest PyTorch integer dtype that can hold n_bits values."""
    if n_bits <= 8:
        return torch.int8
    elif n_bits <= 16:
        return torch.int16
    else:
        return torch.int32


def pack_intN(int_tensor, n_bits):
    """
    Pack quantized integer values into bytes for compact storage.
    Works for any bit-width from 1 to 16.

    Strategy:
      - n_bits <= 8 and 8 is divisible by n_bits: pack multiple values per byte
        (e.g. 4-bit -> 2 per byte, 2-bit -> 4 per byte, 1-bit -> 8 per byte)
      - n_bits == 8: store as int8 directly (1 byte per value)
      - n_bits > 8 (e.g. 16): store as int16 (2 bytes per value)
      - otherwise (e.g. 3, 5, 6, 7): store each value in the smallest
        standard dtype that fits (int8 for <=8 bits, int16 for <=16 bits).
        This wastes some bits per element but keeps the code simple.

    Signed values are shifted to unsigned range before packing.

    Args:
        int_tensor: tensor with integer values (any dtype, will be cast)
        n_bits: bit-width (1 to 16)
    Returns:
        (packed_tensor, original_numel)
    """
    assert 1 <= n_bits <= 16, f"n_bits must be in [1, 16], got {n_bits}"
    flat = int_tensor.flatten().to(torch.int32)
    numel = flat.numel()

    # Shift to unsigned range for packing
    shift = (1 << (n_bits - 1)) - 1
    flat_unsigned = flat + shift  # values now in [0, 2^n_bits - 2]

    if n_bits > 8:
        # Store as int16 (2 bytes per value)
        return flat.to(torch.int16), numel

    if n_bits == 8:
        return flat.to(torch.int8), numel

    # n_bits < 8: try to bit-pack if 8 is evenly divisible by n_bits
    if 8 % n_bits == 0:
        vals_per_byte = 8 // n_bits
        flat_u8 = flat_unsigned.to(torch.uint8)

        # Pad to multiple of vals_per_byte
        pad = (vals_per_byte - numel % vals_per_byte) % vals_per_byte
        if pad > 0:
            flat_u8 = torch.cat([flat_u8, torch.zeros(pad, dtype=torch.uint8)])

        # Pack vals_per_byte values into each byte
        flat_u8 = flat_u8.reshape(-1, vals_per_byte)
        packed = torch.zeros(flat_u8.shape[0], dtype=torch.uint8)
        for i in range(vals_per_byte):
            packed |= flat_u8[:, i] << (i * n_bits)

        return packed, numel
    else:
        # Non-divisible (3, 5, 6, 7 bits): store in int8 (wastes some bits, but correct)
        return flat.to(torch.int8), numel


def unpack_intN(packed, n_bits, numel, signed=True):
    """
    Unpack bytes back to quantized integer values.

    Args:
        packed: tensor from pack_intN
        n_bits: bit-width (1 to 16)
        numel: original number of elements (before padding)
        signed: if True, shift back to signed range
    Returns:
        float tensor with integer values
    """
    assert 1 <= n_bits <= 16

    if n_bits > 8:
        # Was stored as int16
        return packed.float()[:numel]

    if n_bits == 8:
        return packed.float()[:numel]

    if 8 % n_bits == 0:
        # Was bit-packed
        vals_per_byte = 8 // n_bits
        mask = (1 << n_bits) - 1
        packed = packed.to(torch.uint8)

        unpacked_parts = []
        for i in range(vals_per_byte):
            unpacked_parts.append(((packed >> (i * n_bits)) & mask).float())

        # Stack and flatten to original order
        result = torch.stack(unpacked_parts, dim=1).flatten()[:numel]

        if signed:
            shift = (1 << (n_bits - 1)) - 1
            result = result - shift
        return result
    else:
        # Was stored as int8 (non-divisible bit-widths)
        return packed.float()[:numel]


# ============================================================
# Activation fake-quant module (with calibration / freeze)
# ============================================================

class ActFakeQuant(nn.Module):
    """
    Per-tensor activation fake-quant with configurable bits.
    - In calibration mode (frozen=False): observes min/max of activations.
    - After freeze(): applies fake quantization (quantize + dequantize).
    - In QAT mode (qat=True): uses STE for gradient flow.
    Activations are always computed on-the-fly, so no int storage is needed.
    """
    def __init__(self, n_bits=8, unsigned=True, qat=False):
        super().__init__()
        self.n_bits = n_bits
        self.unsigned = unsigned
        self.qat = qat
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def observe(self, x):
        """Update running min/max from observed activations."""
        self.min_val = torch.minimum(self.min_val, x.min())
        self.max_val = torch.maximum(self.max_val, x.max())

    @torch.no_grad()
    def freeze(self):
        """Finalize scale and zero-point from observed min/max."""
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
        if self.qat:
            return ste_fake_quantize(x, self.scale, self.zp, self.qmin, self.qmax)
        else:
            return fake_quantize(x, self.scale, self.zp, self.qmin, self.qmax)

    def extra_repr(self):
        return f"n_bits={self.n_bits}, unsigned={self.unsigned}, frozen={self.frozen}, qat={self.qat}"


# ============================================================
# Weight quantization wrappers (Conv2d and Linear)
# ============================================================
# Three modes of operation:
#   1. No quantization (frozen=False, qat=False): vanilla fp32 forward.
#   2. Fake-quant for QAT (qat=True): STE fake-quant, recomputes scale each step.
#   3. Fake-quant for PTQ (frozen=True, qat=False): uses frozen scale/zp.
#   4. Int-weight inference (_int_weight is not None): dequantize int8 -> fp32, then MAC.

class QuantConv2d(nn.Conv2d):
    """Conv2d with per-tensor symmetric weight quantization."""

    def __init__(self, *args, weight_bits=8, qat=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.qat = qat
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None
        # Buffer for actual int weight storage (set by quantize_weight_to_int)
        self.register_buffer("_int_weight", None)

    @torch.no_grad()
    def freeze(self):
        """Compute and freeze weight quantization parameters from current fp32 weights."""
        w = self.weight.detach()
        w_min, w_max = w.min(), w.max()
        scale, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.w_scale.copy_(scale)
        self.w_zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    @torch.no_grad()
    def quantize_weight_to_int(self):
        """Convert fp32 weight to actual intN values. Call after freeze()."""
        assert self.frozen, "Must call freeze() before quantize_weight_to_int()"
        int_w = quantize(self.weight.data, self.w_scale, self.w_zp, self.qmin, self.qmax)
        self._int_weight = int_w.to(_int_dtype_for_bits(self.weight_bits))
        # Zero out fp32 weight - no longer needed for forward
        self.weight.requires_grad_(False)
        self.weight.data.zero_()

    def _get_quantized_weight(self):
        """Fake-quant path (used during QAT training or PTQ evaluation before int conversion)."""
        if self.qat:
            w = self.weight
            w_min, w_max = w.min().detach(), w.max().detach()
            scale, zp, qmin, qmax = qparams_from_minmax(
                w_min, w_max, n_bits=self.weight_bits, unsigned=False
            )
            return ste_fake_quantize(w, scale, zp, qmin, qmax)
        else:
            return fake_quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)

    def forward(self, x):
        if self._int_weight is not None:
            # Dequant-only inference: intN -> float -> MAC
            w_fp32 = dequantize(self._int_weight.float(), self.w_scale, self.w_zp)
            return F.conv2d(x, w_fp32, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if not self.frozen and not self.qat:
            # No quantization at all
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # Fake-quant path (QAT or PTQ before int conversion)
        w_dq = self._get_quantized_weight()
        return F.conv2d(x, w_dq, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        base = super().extra_repr()
        int_mode = self._int_weight is not None
        return base + f", weight_bits={self.weight_bits}, qat={self.qat}, int_mode={int_mode}"


class QuantLinear(nn.Linear):
    """Linear with per-tensor symmetric weight quantization."""

    def __init__(self, *args, weight_bits=8, qat=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.qat = qat
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None
        self.register_buffer("_int_weight", None)

    @torch.no_grad()
    def freeze(self):
        w = self.weight.detach()
        w_min, w_max = w.min(), w.max()
        scale, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.w_scale.copy_(scale)
        self.w_zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    @torch.no_grad()
    def quantize_weight_to_int(self):
        """Convert fp32 weight to actual intN values. Call after freeze()."""
        assert self.frozen, "Must call freeze() before quantize_weight_to_int()"
        int_w = quantize(self.weight.data, self.w_scale, self.w_zp, self.qmin, self.qmax)
        self._int_weight = int_w.to(_int_dtype_for_bits(self.weight_bits))
        self.weight.requires_grad_(False)
        self.weight.data.zero_()

    def _get_quantized_weight(self):
        if self.qat:
            w = self.weight
            w_min, w_max = w.min().detach(), w.max().detach()
            scale, zp, qmin, qmax = qparams_from_minmax(
                w_min, w_max, n_bits=self.weight_bits, unsigned=False
            )
            return ste_fake_quantize(w, scale, zp, qmin, qmax)
        else:
            return fake_quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)

    def forward(self, x):
        if self._int_weight is not None:
            w_fp32 = dequantize(self._int_weight.float(), self.w_scale, self.w_zp)
            return F.linear(x, w_fp32, self.bias)
        if not self.frozen and not self.qat:
            return F.linear(x, self.weight, self.bias)
        w_dq = self._get_quantized_weight()
        return F.linear(x, w_dq, self.bias)

    def extra_repr(self):
        base = super().extra_repr()
        int_mode = self._int_weight is not None
        return base + f", weight_bits={self.weight_bits}, qat={self.qat}, int_mode={int_mode}"


# ============================================================
# Model surgery: swap layers with quantized versions
# ============================================================

def swap_to_quant_modules(model, weight_bits=8, act_bits=8,
                          activations_unsigned=True, qat=False,
                          skip_layers=None, _prefix=""):
    """
    Replace Conv2d/Linear with Quant* versions (weight fake-quant),
    and insert ActFakeQuant after ReLU/ReLU6 activations.

    Args:
        model: nn.Module to modify in-place.
        weight_bits: bit-width for weight quantization.
        act_bits: bit-width for activation quantization.
        activations_unsigned: True for post-ReLU (unsigned), False for signed.
        qat: if True, enable STE for gradient flow (QAT mode).
        skip_layers: set of layer name prefixes to skip quantization on.
                     e.g. {"features.0", "classifier"} to skip first conv and classifier.
        _prefix: internal use for tracking the full module path.
    """
    if skip_layers is None:
        skip_layers = set()

    for name, m in list(model.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name

        # Recurse into children first
        swap_to_quant_modules(m, weight_bits, act_bits, activations_unsigned, qat, skip_layers, _prefix=full_name)

        # Check if this layer (or any parent) should be skipped
        should_skip = any(
            full_name == s or full_name.startswith(s + ".")
            for s in skip_layers
        )
        if should_skip:
            continue

        if isinstance(m, nn.Conv2d) and not isinstance(m, QuantConv2d):
            q = QuantConv2d(
                m.in_channels, m.out_channels, m.kernel_size,
                stride=m.stride, padding=m.padding, dilation=m.dilation,
                groups=m.groups, bias=(m.bias is not None),
                weight_bits=weight_bits, qat=qat
            )
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.Linear) and not isinstance(m, QuantLinear):
            q = QuantLinear(m.in_features, m.out_features, bias=(m.bias is not None), weight_bits=weight_bits, qat=qat)
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, (nn.ReLU, nn.ReLU6)):
            seq = nn.Sequential(OrderedDict([
                ("act", nn.ReLU6(inplace=False)),
                ("aq", ActFakeQuant(n_bits=act_bits, unsigned=activations_unsigned, qat=qat)),
            ]))
            setattr(model, name, seq)


def swap_single_layer_quant(model, target_layer_name, weight_bits=None,
                            act_bits=None, qat=False):
    """
    Quantize only a single named layer (for sensitivity analysis).
    target_layer_name is a dot-separated path like 'features.1.conv.0.0'.
    """
    parts = target_layer_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    attr_name = parts[-1]
    m = getattr(parent, attr_name) if not attr_name.isdigit() else parent[int(attr_name)]

    if weight_bits is not None and isinstance(m, nn.Conv2d) and not isinstance(m, QuantConv2d):
        q = QuantConv2d(
            m.in_channels, m.out_channels, m.kernel_size,
            stride=m.stride, padding=m.padding, dilation=m.dilation,
            groups=m.groups, bias=(m.bias is not None),
            weight_bits=weight_bits, qat=qat
        )
        q.weight.data.copy_(m.weight.data)
        if m.bias is not None:
            q.bias.data.copy_(m.bias.data)
        if attr_name.isdigit():
            parent[int(attr_name)] = q
        else:
            setattr(parent, attr_name, q)

    elif weight_bits is not None and isinstance(m, nn.Linear) and not isinstance(m, QuantLinear):
        q = QuantLinear(
            m.in_features, m.out_features,
            bias=(m.bias is not None),
            weight_bits=weight_bits, qat=qat
        )
        q.weight.data.copy_(m.weight.data)
        if m.bias is not None:
            q.bias.data.copy_(m.bias.data)
        if attr_name.isdigit():
            parent[int(attr_name)] = q
        else:
            setattr(parent, attr_name, q)

    elif act_bits is not None and isinstance(m, (nn.ReLU, nn.ReLU6)):
        seq = nn.Sequential(OrderedDict([
            ("act", nn.ReLU6(inplace=False)),
            ("aq", ActFakeQuant(n_bits=act_bits, unsigned=True, qat=qat)),
        ]))
        if attr_name.isdigit():
            parent[int(attr_name)] = seq
        else:
            setattr(parent, attr_name, seq)


# ============================================================
# Freeze / calibrate / quantize helpers
# ============================================================

def freeze_all_quant(model):
    """Freeze all quantization parameters (weights + activations), (finalize scales/ZPs) after calibration."""
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.freeze()
        if isinstance(mod, ActFakeQuant):
            mod.freeze()


def quantize_model_weights(model):
    """Convert all QuantConv2d/QuantLinear fp32 weights to actual int8.
    Call after freeze_all_quant(). After this, forward() uses dequant-only path."""
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.quantize_weight_to_int()


def extract_original_weights(model):
    """
    Extract only the original model weight/bias parameters from a quantized model,
    stripping out quantization buffers (w_scale, w_zp, min_val, max_val, scale, zp).
    This allows loading the state dict back into a vanilla (non-quantized) model.
    """
    quant_buffer_names = {"w_scale", "w_zp", "min_val", "max_val", "scale", "zp", "_int_weight"}
    clean_state_dict = {}
    for key, val in model.state_dict().items():
        param_name = key.split(".")[-1]
        if param_name in quant_buffer_names:
            continue
        if ".aq." in key or ".act." in key:
            continue
        clean_state_dict[key] = val
    return clean_state_dict


def calibrate(model, dataloader, n_batches, device=None):
    """Run calibration: forward pass on n_batches to observe activation ranges."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    count = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data[0].to(device)
            model(inputs)
            count += 1
            if count >= n_batches:
                break


# ============================================================
# Save / Load quantized model (actual intN storage on disk)
# ============================================================

def save_quantized_model(model, path, weight_bits, act_bits):
    """
    Save model with weights packed as actual intN bytes.
    The saved file contains:
      - Packed intN weights (genuinely small)
      - Per-layer scale, zero_point, bias (fp32)
      - Per-activation scale, zero_point
      - Metadata (bit-widths, layer shapes)
    """
    data = {
        "weight_bits": weight_bits,
        "act_bits": act_bits,
        "layers": {},
    }

    for name, mod in model.named_modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            # Get int weight values (either from _int_weight buffer or by quantizing)
            if mod._int_weight is not None:
                int_w = mod._int_weight.cpu().flatten().float()
            else:
                assert mod.frozen, f"Layer {name} not frozen. Call freeze_all_quant() first."
                int_w = quantize(mod.weight.data, mod.w_scale, mod.w_zp, mod.qmin, mod.qmax).cpu().flatten()

            # Pack into bytes
            packed, numel = pack_intN(int_w, weight_bits)

            layer_data = {
                "type": "conv2d" if isinstance(mod, QuantConv2d) else "linear",
                "packed_weight": packed,
                "weight_shape": list(mod.weight.shape),
                "weight_numel": numel,
                "w_scale": mod.w_scale.cpu().clone(),
                "w_zp": mod.w_zp.cpu().clone(),
                "qmin": mod.qmin,
                "qmax": mod.qmax,
                "weight_bits": mod.weight_bits,
            }
            if mod.bias is not None:
                layer_data["bias"] = mod.bias.data.cpu().clone()
            data["layers"][name] = layer_data

        elif isinstance(mod, ActFakeQuant) and mod.frozen:
            data["layers"][name] = {
                "type": "act_fake_quant",
                "scale": mod.scale.cpu().clone(),
                "zp": mod.zp.cpu().clone(),
                "n_bits": mod.n_bits,
                "qmin": mod.qmin,
                "qmax": mod.qmax,
                "unsigned": mod.unsigned,
            }

        elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            data["layers"][name] = {
                "type": "batchnorm",
                "state_dict": {k: v.cpu().clone() for k, v in mod.state_dict().items()},
            }

    torch.save(data, path)

    file_size = os.path.getsize(path)
    print(f"Saved quantized model to {path}")
    print(f"  File size on disk: {file_size / 1024 / 1024:.4f} MB")


def load_quantized_model(path, model):
    """
    Load packed intN weights into a model that already has
    QuantConv2d / QuantLinear / ActFakeQuant modules (from swap_to_quant_modules).

    After loading, each weight layer holds actual int8 weights and
    forward() uses the dequant-only path.
    """
    data = torch.load(path, map_location="cpu")
    weight_bits = data["weight_bits"]

    for name, mod in model.named_modules():
        if name not in data["layers"]:
            continue
        info = data["layers"][name]

        if isinstance(mod, (QuantConv2d, QuantLinear)):
            # Unpack int weights
            int_flat = unpack_intN(
                info["packed_weight"], weight_bits, info["weight_numel"], signed=True
            )
            shape = info["weight_shape"]
            int_tensor = int_flat.reshape(shape).to(_int_dtype_for_bits(weight_bits))

            # Store as intN buffer and set frozen params
            mod._int_weight = int_tensor
            mod.w_scale.copy_(info["w_scale"])
            mod.w_zp.copy_(info["w_zp"])
            mod.qmin = info["qmin"]
            mod.qmax = info["qmax"]
            mod.frozen = True

            # Zero out fp32 weight (not needed, intN is used for forward)
            mod.weight.requires_grad_(False)
            mod.weight.data.zero_()

            if "bias" in info:
                mod.bias.data.copy_(info["bias"])

        elif isinstance(mod, ActFakeQuant):
            mod.scale.copy_(info["scale"])
            mod.zp.copy_(info["zp"])
            mod.qmin = info["qmin"]
            mod.qmax = info["qmax"]
            mod.frozen = True

        elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.load_state_dict(info["state_dict"])

    print(f"Loaded quantized model from {path} (weight_bits={weight_bits})")


# ============================================================
# Model size estimation utilities
# ============================================================

def model_size_bytes_fp32(model):
    """Total size of all parameters stored as FP32 (4 bytes each)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * 4
    return total


def model_size_bytes_quant(model, weight_bits=8):
    """
    Estimated size if weights are stored at weight_bits, biases stay FP32.
    Also accounts for per-tensor scale+zp overhead (8 bytes per quantized layer).
    """
    total_bytes = 0
    overhead_bytes = 0

    for name, p in model.named_parameters():
        if "weight" in name:
            total_bytes += p.numel() * weight_bits / 8
        elif "bias" in name:
            total_bytes += p.numel() * 4  # biases stay FP32
        else:
            total_bytes += p.numel() * 4  # other params stay FP32

    # Count quantized layers for overhead
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            overhead_bytes += 8  # scale (4B) + zero_point (4B)
        if isinstance(mod, ActFakeQuant) and mod.frozen:
            overhead_bytes += 8  # scale + zp for activation

    return total_bytes, overhead_bytes


def activation_size_bytes(model, input_tensor, act_bits=8):
    """
    Measure total activation memory during a forward pass.
    Returns (fp32_bytes, quantized_bytes, overhead_bytes).
    """
    activation_numel = []
    hooks = []

    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            activation_numel.append(out.numel())

    for mod in model.modules():
        if isinstance(mod, (nn.ReLU, nn.ReLU6, ActFakeQuant)):
            hooks.append(mod.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    total_numel = sum(activation_numel)
    fp32_bytes = total_numel * 4
    quant_bytes = total_numel * act_bits / 8
    overhead = len(activation_numel) * 8

    return fp32_bytes, quant_bytes, overhead


def print_compression(model, weight_bits=8, act_bits=8, input_tensor=None):
    """Print compression summary for weights and activations."""
    fp32_size = model_size_bytes_fp32(model)
    quant_size, quant_overhead = model_size_bytes_quant(model, weight_bits)
    total_quant = quant_size + quant_overhead
    ratio = fp32_size / max(total_quant, 1)

    print("=" * 50)
    print("Compression Summary")
    print("=" * 50)
    print(f"FP32 model size:       {fp32_size / 1024 / 1024:.4f} MB")
    print(f"Quantized weight size: {quant_size / 1024 / 1024:.4f} MB (weights={weight_bits}-bit)")
    print(f"Quantization overhead: {quant_overhead / 1024:.4f} KB (scales + zero points)")
    print(f"Total quantized size:  {total_quant / 1024 / 1024:.4f} MB")
    print(f"Weight compression:    {ratio:.2f}x")

    if input_tensor is not None:
        fp32_act, quant_act, act_overhead = activation_size_bytes(model, input_tensor, act_bits)
        act_ratio = fp32_act / max(quant_act + act_overhead, 1)
        print(f"FP32 activation size:  {fp32_act / 1024 / 1024:.4f} MB (single input)")
        print(f"Quant activation size: {quant_act / 1024 / 1024:.4f} MB (act={act_bits}-bit)")
        print(f"Activation overhead:   {act_overhead / 1024:.4f} KB")
        print(f"Activation compress.:  {act_ratio:.2f}x")

    print("=" * 50)

    result = {
        "fp32_model_size_mb": fp32_size / 1024 / 1024,
        "quant_model_size_mb": total_quant / 1024 / 1024,
        "weight_compression_ratio": ratio,
        "overhead_kb": quant_overhead / 1024,
    }
    if input_tensor is not None:
        result["fp32_activation_size_mb"] = fp32_act / 1024 / 1024
        result["quant_activation_size_mb"] = (quant_act + act_overhead) / 1024 / 1024
        result["activation_compression_ratio"] = act_ratio
    return result


# ============================================================
# Helper: list quantizable layers (for sensitivity analysis)
# ============================================================

def get_quantizable_layers(model):
    """
    Return list of (name, module) for all Conv2d and Linear layers,
    plus (name, module) for all ReLU/ReLU6 activation layers.
    """
    weight_layers = []
    act_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, QuantConv2d)):
            weight_layers.append((name, mod))
        elif isinstance(mod, (nn.Linear, QuantLinear)):
            weight_layers.append((name, mod))
        elif isinstance(mod, (nn.ReLU, nn.ReLU6)):
            act_layers.append((name, mod))
    return weight_layers, act_layers
