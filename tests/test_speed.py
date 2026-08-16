from __future__ import annotations

from easytrain.core.speed import Hardware, plan_speed


def _gpu(name: str, capability: tuple[int, int], vram_gb: float) -> Hardware:
    return Hardware(
        device="cuda",
        name=name,
        capability=capability,
        vram_bytes=int(vram_gb * 1024**3),
    )


def test_ampere_uses_bf16_sdpa_fused():
    plan = plan_speed(_gpu("NVIDIA A100-SXM4-40GB", (8, 0), 40), n_params=66_000_000)
    assert plan.precision == "bf16"
    assert plan.attn_implementation == "sdpa"
    assert plan.optim == "adamw_torch_fused"
    assert plan.tf32 is True
    assert plan.torch_compile is False
    assert "BF16" in plan.why_fast or "bf16" in plan.why_fast.lower() or "A100" in plan.why_fast


def test_t4_uses_fp16():
    plan = plan_speed(_gpu("Tesla T4", (7, 5), 16), n_params=66_000_000)
    assert plan.precision == "fp16"
    assert plan.tf32 is False


def test_cpu_is_fp32_no_fused():
    plan = plan_speed(Hardware(device="cpu", name="CPU", capability=None, vram_bytes=None))
    assert plan.precision == "fp32"
    assert plan.optim == "adamw_torch"
    assert plan.pin_memory is False
    assert "CPU" in plan.why_fast


def test_stable_does_not_compile():
    plan = plan_speed(_gpu("NVIDIA RTX 4090", (8, 9), 24), speed="stable")
    assert plan.torch_compile is False


def test_max_compiles_on_cuda():
    plan = plan_speed(_gpu("NVIDIA RTX 4090", (8, 9), 24), speed="max")
    assert plan.torch_compile is True


def test_peft_auto_full_finetune_for_small_encoder():
    plan = plan_speed(_gpu("NVIDIA RTX 4090", (8, 9), 24), peft="auto", n_params=66_000_000)
    assert plan.peft == "none"


def test_peft_auto_lora_for_billion_param():
    plan = plan_speed(_gpu("NVIDIA RTX 4090", (8, 9), 24), peft="auto", n_params=7_000_000_000)
    assert plan.peft == "lora"


def test_explicit_batch_size():
    plan = plan_speed(
        Hardware(device="cpu", name="CPU", capability=None, vram_bytes=None),
        batch_size=3,
    )
    assert plan.batch_size == 3
