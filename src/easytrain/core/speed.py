from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Hardware:
    device: str
    name: str
    capability: tuple[int, int] | None
    vram_bytes: int | None

    @property
    def vram_gb(self) -> float | None:
        if self.vram_bytes is None:
            return None
        return self.vram_bytes / (1024**3)


@dataclass(frozen=True)
class SpeedPlan:
    precision: str
    torch_dtype_name: str | None
    attn_implementation: str | None
    batch_size: int
    optim: str
    tf32: bool
    pin_memory: bool
    num_workers: int
    gradient_checkpointing: bool
    torch_compile: bool
    peft: str
    why_fast: str
    estimated_vram_gb: float | None
    fallbacks: tuple[str, ...] = ()

    @property
    def torch_dtype(self):
        if self.torch_dtype_name == "bfloat16":
            return torch.bfloat16
        if self.torch_dtype_name == "float16":
            return torch.float16
        return None


def detect_hardware() -> Hardware:
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        return Hardware(
            device="cuda",
            name=getattr(props, "name", "CUDA GPU"),
            capability=capability,
            vram_bytes=int(getattr(props, "total_memory", 0) or 0),
        )
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return Hardware(device="mps", name="Apple MPS", capability=None, vram_bytes=None)
    return Hardware(device="cpu", name="CPU", capability=None, vram_bytes=None)


def _generation(capability: tuple[int, int] | None) -> str:
    if capability is None:
        return "none"
    major, _minor = capability
    if major >= 9:
        return "hopper+"
    if major >= 8:
        return "ampere+"
    if major >= 7:
        return "volta/turing"
    return "older"


def _pick_precision(hardware: Hardware, speed: str) -> str:
    if hardware.device == "cpu":
        return "fp32"
    if hardware.device == "mps":
        return "fp16"
    generation = _generation(hardware.capability)
    if generation in {"ampere+", "hopper+"}:
        return "bf16"
    return "fp16"


def _pick_batch_size(
    hardware: Hardware,
    n_params: int | None,
    requested: int | str,
) -> int:
    if isinstance(requested, int):
        return max(1, requested)
    if hardware.device == "cpu":
        return 8 if (n_params or 0) < 200_000_000 else 2
    if hardware.device == "mps":
        return 16
    vram = hardware.vram_gb or 8.0
    params = n_params or 66_000_000
    if params >= 1_000_000_000:
        if vram < 16:
            return 1
        if vram < 24:
            return 2
        return 4
    if vram < 6:
        return 8
    if vram < 12:
        return 16
    if vram < 20:
        return 32
    return 64


def _pick_peft(requested: bool | str, n_params: int | None, hardware: Hardware) -> str:
    if requested in {False, "none", "full", None}:
        return "none"
    if requested in {True, "lora"}:
        return "lora"
    if requested == "qlora":
        return "qlora"
    params = n_params or 0
    if params >= 1_000_000_000:
        return "lora"
    vram = hardware.vram_gb
    if vram is not None and vram < 6 and params > 300_000_000:
        return "lora"
    return "none"


def estimate_vram_gb(
    n_params: int | None,
    batch_size: int,
    hidden_size: int | None,
    num_layers: int | None,
    precision: str,
    seq_len: int = 128,
) -> float | None:
    if not n_params:
        return None
    dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}.get(precision, 4)
    weights = n_params * dtype_bytes
    grads = n_params * dtype_bytes
    adam = n_params * 8
    hidden = hidden_size or 768
    layers = num_layers or 12
    activations = batch_size * seq_len * hidden * layers * 8
    return (weights + grads + adam + activations) / (1024**3)


def estimate_params(config: Any | None) -> int | None:
    if config is None:
        return None
    hidden = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
    layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layers", None)
    vocab = getattr(config, "vocab_size", None)
    intermediate = getattr(config, "intermediate_size", None)
    if not hidden or not layers:
        return None
    intermediate = intermediate or hidden * 4
    vocab = vocab or 30522
    embeddings = vocab * hidden
    attn = 4 * hidden * hidden
    ff = 2 * hidden * intermediate
    return int(embeddings + layers * (attn + ff))


def plan_speed(
    hardware: Hardware,
    *,
    speed: str = "auto",
    batch_size: int | str = "auto",
    peft: bool | str = "auto",
    n_params: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> SpeedPlan:
    if speed not in {"auto", "stable", "max"}:
        from easytrain.errors import ConfigError

        raise ConfigError("speed must be 'auto', 'stable', or 'max'.")

    precision = _pick_precision(hardware, speed)
    torch_dtype_name = {"bf16": "bfloat16", "fp16": "float16", "fp32": None}[precision]
    resolved_batch = _pick_batch_size(hardware, n_params, batch_size)
    resolved_peft = _pick_peft(peft, n_params, hardware)
    generation = _generation(hardware.capability)
    tf32 = hardware.device == "cuda" and generation in {"ampere+", "hopper+"}
    fused = hardware.device == "cuda"
    attn = "sdpa" if hardware.device in {"cuda", "cpu", "mps"} else None
    compile_on = speed == "max" and hardware.device == "cuda"
    checkpoint = bool(hardware.vram_gb is not None and hardware.vram_gb < 8 and (n_params or 0) > 80_000_000)
    workers = 2 if hardware.device == "cuda" else 0
    pin = hardware.device == "cuda"
    vram = estimate_vram_gb(n_params, resolved_batch, hidden_size, num_layers, precision)
    fallbacks: list[str] = []
    if speed == "max":
        fallbacks.append("v1 encoder path: torch.compile only; FP8/Liger are LLM-later")
        if not compile_on:
            fallbacks.append("torch.compile skipped (no CUDA)")

    why = _why_fast(
        hardware,
        precision=precision,
        attn=attn,
        fused=fused,
        batch=resolved_batch,
        compile_on=compile_on,
    )
    return SpeedPlan(
        precision=precision,
        torch_dtype_name=torch_dtype_name,
        attn_implementation=attn,
        batch_size=resolved_batch,
        optim="adamw_torch_fused" if fused else "adamw_torch",
        tf32=tf32,
        pin_memory=pin,
        num_workers=workers,
        gradient_checkpointing=checkpoint,
        torch_compile=compile_on,
        peft=resolved_peft,
        why_fast=why,
        estimated_vram_gb=vram,
        fallbacks=tuple(fallbacks),
    )


def _why_fast(
    hardware: Hardware,
    *,
    precision: str,
    attn: str | None,
    fused: bool,
    batch: int,
    compile_on: bool,
) -> str:
    if hardware.device == "cpu":
        return "CPU fp32; no CUDA. Install a CUDA build of PyTorch for BF16, SDPA on GPU, and fused AdamW."
    if hardware.device == "mps":
        return f"Apple MPS with {precision}; SDPA attention, batch {batch}."
    bits = [f"{precision.upper()} on {hardware.name}"]
    if attn:
        bits.append(f"{attn.upper()} attention")
    if fused:
        bits.append("fused AdamW")
    bits.append(f"batch {batch}")
    if hardware.vram_gb:
        bits.append(f"~{hardware.vram_gb:.0f} GB VRAM")
    if compile_on:
        bits.append("torch.compile")
    return " + ".join(bits) + "."


def apply_runtime_flags(speed: SpeedPlan, hardware: Hardware) -> None:
    if hardware.device == "cuda" and speed.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def maybe_compile(model: Any, speed: SpeedPlan) -> tuple[Any, str | None]:
    if not speed.torch_compile:
        return model, None
    try:
        compiled = torch.compile(model)
        return compiled, None
    except Exception as exc:
        return model, f"torch.compile failed ({exc}); continuing without it"
