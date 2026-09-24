"""Hardware-aware autocast; FP16 generator training requires GradScaler."""
from contextlib import nullcontext
import torch


def amp_dtype(device, precision="auto", use_bf16=True):
    device = torch.device(device)
    if precision not in ("auto", "fp32", "fp16", "bf16"):
        raise ValueError(f"Unknown precision: {precision}")
    if device.type != "cuda" or precision == "fp32":
        return None
    native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
    if precision == "bf16":
        if not native_bf16:
            raise ValueError("BF16 requires native support (Ampere or newer); use auto or fp16 on T4.")
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if not use_bf16:
        return None
    return torch.bfloat16 if native_bf16 else torch.float16


def autocast_context(device, dtype):
    return torch.autocast(torch.device(device).type, dtype=dtype) if dtype else nullcontext()


def potential_autocast(device, use_bf16):
    # Potential input gradients and small importance-weight differences need
    # FP32 on T4. Do not silently run unsupported BF16 or unscaled FP16 here.
    native = torch.device(device).type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    # A nullcontext would inherit the generator's ambient autocast. Explicitly
    # disable it when this head requests FP32, including its cond projection.
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                          enabled=bool(use_bf16 and native))
