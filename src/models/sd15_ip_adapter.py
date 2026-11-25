# Phạm vi: baseline và proposed dùng DDIM; Proposed thêm ảnh tham chiếu qua IP-Adapter và callback trộn latent.
"""SD1.5 inpainting hooks for DDIM blending and optional IP-Adapter.

Latent blending is restricted to DDIM, whose post-step noise level is explicit.
Image-prompt attention requires real compatible adapter weights, not just embeds.
"""
from __future__ import annotations

import inspect
import math
import numpy as np

SD15_INPAINTING = "stable-diffusion-v1-5/stable-diffusion-inpainting"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_PLUS = "ip-adapter-plus_sd15.safetensors"


# Kiểm tra backbone, cờ, ngưỡng và tham số chạy chung của proposed.
def validate_proposed_config(config):
    if config.get("pipeline_mode") != "proposed":
        raise ValueError("Expected proposed config")
    if config.get("inpainting_model_id") != SD15_INPAINTING:
        raise ValueError("All experiments require the SD1.5 inpainting checkpoint")
    for key in ("use_micro_attributes", "use_ip_adapter", "use_latent_blending", "use_strict_critic", "enable_critic"):
        if type(config.get(key)) is not bool:
            raise ValueError(f"{key} must be a JSON boolean")
    attempts = config.get("max_critic_iter")
    if type(attempts) is not int or not 1 <= attempts <= 10:
        raise ValueError("max_critic_iter counts TOTAL attempts and must be 1..10")
    if config.get("refinement_scheduler") != "ddim":
        raise ValueError("Proposed requires refinement_scheduler='ddim'")
    for key, low, high in (("critique_threshold", 0, 10), ("ip_adapter_scale", 0, 1),
                           ("guidance_scale", 1, 9), ("retry_guidance_step", 0, 3)):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be in [{low}, {high}]")
    steps = config.get("num_inference_steps", 28)
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("num_inference_steps must be an integer in [1, 1000]")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if config.get("use_ip_adapter"):
        if config.get("ip_adapter_model_id") != IP_ADAPTER_REPO or config.get("ip_adapter_weight_name") != IP_ADAPTER_PLUS:
            raise ValueError("Proposed uses the official h94 IP-Adapter Plus SD1.5 safetensors checkpoint")


# Dùng cùng scheduler DDIM để phép so sánh không đổi lịch khử nhiễu giữa các phiên bản.
def configure_ddim(pipe):
    from diffusers import DDIMScheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="leading",
                                               steps_offset=0, set_alpha_to_one=True, clip_sample=False)
    return pipe


# Kiểm tra loại UNet, lắp DDIM và chỉ nạp IP-Adapter khi bật ở Proposed.
def configure_proposed_pipe(pipe, config):
    """Called on CPU at initial load, before model residency management moves to CUDA."""
    validate_proposed_config(config)
    supported = inspect.signature(pipe.__call__).parameters
    if "callback_on_step_end" not in supported or "ip_adapter_image" not in supported:
        raise RuntimeError("Incompatible diffusers API; rerun Colab setup with requirements-colab.txt")
    if pipe.unet.config.in_channels != 9:
        raise ValueError("Proposed requires a dedicated 9-channel inpainting UNet")
    # Preserve the checkpoint beta schedule and prediction settings.
    configure_ddim(pipe)
    if config["use_ip_adapter"]:
        if pipe.unet.config.cross_attention_dim != 768:
            raise ValueError("SD1.5 IP-Adapter requires cross_attention_dim=768")
        print("[Models] Loading IP-Adapter Plus SD1.5 and CLIP ViT-H image encoder...", flush=True)
        pipe.load_ip_adapter(config["ip_adapter_model_id"], subfolder="models",
                             weight_name=config["ip_adapter_weight_name"], image_encoder_folder="image_encoder")
        if pipe.image_encoder is None or pipe.unet.encoder_hid_proj is None:
            raise RuntimeError("IP-Adapter loading did not install the image encoder/projection")
        pipe.set_ip_adapter_scale(config["ip_adapter_scale"])
    return pipe


# Xác định mức nhiễu của latent tham chiếu sau bước scheduler, không dùng nhầm mức trước bước.
def ddim_previous_timestep(scheduler, timestep):
    """Noise level AFTER scheduler.step, including strength-truncated schedules."""
    return int(timestep) - int(scheduler.config.num_train_timesteps) // int(scheduler.num_inference_steps)


# Giữ latent sinh ở vùng được sửa, lấy latent tham chiếu ở vùng còn lại.
def blend_latents(generated, reference, editable):
    """editable=1 keeps generation; editable=0 restores the reference latent."""
    return editable * generated + (1 - editable) * reference


# Callback theo từng bước denoise của Proposed.
class ReferenceLatentBlender:
    # Giữ latent sạch, nhiễu cố định và mask ở độ phân giải latent.
    def __init__(self, clean_latents, noise, editable_mask):
        self.clean = clean_latents
        self.noise = noise
        self.editable = editable_mask
        self.steps = []

    # Tại mỗi bước, giữ latent sinh trong mask và thay phần còn lại bằng tham chiếu đúng mức
    # nhiễu.
    def __call__(self, pipe, step, timestep, tensors):
        import torch
        prev = ddim_previous_timestep(pipe.scheduler, timestep)
        if prev >= 0:
            t = torch.tensor([prev], device=self.clean.device, dtype=torch.long)
            reference = pipe.scheduler.add_noise(self.clean, self.noise, t)
        else:
            # DDIM set_alpha_to_one=True: final target is the clean reference latent.
            reference = self.clean
        current = tensors["latents"]
        if current.shape != reference.shape:
            raise ValueError("Latent blending supports one image per call; unexpected latent shape")
        tensors["latents"] = blend_latents(current, reference.to(current.dtype), self.editable)
        self.steps.append(dict(step=int(step), timestep=int(timestep), reference_timestep=prev))
        return tensors


# Mã hóa seed bằng VAE, đổi mask xuống lưới latent và dùng RNG riêng cho nhiễu tham chiếu.
def make_latent_blender(pipe, image, mask_image, device, seed):
    import torch
    import torch.nn.functional as F
    # image is the clean prior canvas: observed target + generated prior + white.
    # It never contains original RGB from the hidden occluder.
    pixels = pipe.image_processor.preprocess(image).to(device=device, dtype=pipe.vae.dtype)
    clean = pipe.vae.encode(pixels).latent_dist.mode() * pipe.vae.config.scaling_factor
    mask = torch.from_numpy((np.asarray(mask_image) > 0).astype(np.float32))[None, None].to(device)
    # Conservative max pooling retains thin editable regions lost by nearest resize.
    editable = F.adaptive_max_pool2d(mask, clean.shape[-2:]).to(dtype=clean.dtype)
    # A separate, deterministic RNG avoids changing the SD sampling RNG sequence.
    generator = torch.Generator(device=device).manual_seed(int(seed) + 1000003)
    noise = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
    return ReferenceLatentBlender(clean, noise, editable)


# Bọc một lượt chạy Proposed: đưa crop visible vào IP-Adapter và đăng ký callback latent khi có cờ.
class AdvancedInpaintPass:
    """Wrap a single Baseline ROI pass with image conditioning and latent blending.

    A wrapper is constructed per attempt. No callbacks or reference GPU tensors
    are retained by the model registry after a call.
    """
    # Lưu pipeline và crop visible để dùng cho một lượt tạo ứng viên.
    def __init__(self, pipe, config, reference_image, device, seed):
        self.pipe, self.config = pipe, config
        self.reference_image, self.device, self.seed = reference_image, device, seed
        self.diagnostics = {}

    # Bổ sung đối số IP-Adapter và callback nếu bật, sau đó chạy diffusion và lưu chẩn đoán.
    def __call__(self, **kwargs):
        blender = None
        if self.config["use_ip_adapter"]:
            self.pipe.set_ip_adapter_scale(self.config["ip_adapter_scale"])
            kwargs["ip_adapter_image"] = self.reference_image
        if self.config["use_latent_blending"]:
            blender = make_latent_blender(self.pipe, kwargs["image"], kwargs["mask_image"], self.device, self.seed)
            kwargs["callback_on_step_end"] = blender
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        result = self.pipe(**kwargs)
        self.diagnostics = dict(scheduler=type(self.pipe.scheduler).__name__,
                                use_ip_adapter=self.config["use_ip_adapter"],
                                ip_adapter_weight=self.config.get("ip_adapter_weight_name") if self.config["use_ip_adapter"] else None,
                                ip_adapter_scale=self.config["ip_adapter_scale"] if self.config["use_ip_adapter"] else None,
                                use_latent_blending=self.config["use_latent_blending"],
                                num_inference_steps=kwargs["num_inference_steps"],
                                strength=kwargs["strength"],
                                latent_steps=blender.steps if blender else [])
        return result
