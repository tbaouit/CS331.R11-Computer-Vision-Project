"""Real tensor/scheduler tests, also run in Colab. No model weights downloaded."""
import importlib.util
from types import SimpleNamespace
import unittest

AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("diffusers") is not None


@unittest.skipUnless(AVAILABLE, "Requires torch + diffusers (included in Colab environment)")
class LatentCallbackTests(unittest.TestCase):
    def test_tiny_real_inpaint_pipeline_executes_callback_without_weights(self):
        import numpy as np
        import torch
        from PIL import Image
        from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
        from src.models.sd15_ip_adapter import AdvancedInpaintPass, configure_proposed_pipe
        import json
        from pathlib import Path

        torch.manual_seed(7)
        # Random miniature networks test API/shape/device integration, not visual quality.
        unet = UNet2DConditionModel(sample_size=8, in_channels=9, out_channels=4,
            layers_per_block=1, block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=768,
            attention_head_dim=8, norm_num_groups=8)
        vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
            block_out_channels=(32, 64), norm_num_groups=8,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"))
        pipe = StableDiffusionInpaintPipeline(vae=vae, unet=unet, scheduler=DDIMScheduler(steps_offset=1, clip_sample=False),
            text_encoder=None, tokenizer=None, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False).to("cpu")
        pipe.set_progress_bar_config(disable=True)
        cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/proposed.json").read_text())
        cfg["use_ip_adapter"] = False  # no checkpoint download in a unit test
        configure_proposed_pipe(pipe, cfg)
        canvas = Image.new("RGB", (32, 32), (120, 90, 60))
        mask = np.zeros((32, 32), np.uint8)
        mask[8:24, 8:24] = 255
        wrapper = AdvancedInpaintPass(pipe, cfg, canvas, "cpu", 42)
        with torch.inference_mode():
            result = wrapper(image=canvas, mask_image=Image.fromarray(mask),
                prompt_embeds=torch.zeros((1, 4, 768)), height=32, width=32,
                strength=1., num_inference_steps=2, guidance_scale=1.,
                generator=torch.Generator().manual_seed(42))
        self.assertEqual(result.images[0].size, (32, 32))
        self.assertEqual(len(wrapper.diagnostics["latent_steps"]), 2)
        self.assertLess(wrapper.diagnostics["latent_steps"][-1]["reference_timestep"], 0)

        # Install a miniature Plus projection + IP attention + CLIP patch encoder.
        # Exercise the actual image-prompt/CFG path without downloading learned weights.
        from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection
        from diffusers.models.embeddings import IPAdapterPlusImageProjection, MultiIPAdapterImageProjection
        from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
        encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, image_size=32, patch_size=8, projection_dim=32))
        processor = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
        pipe.register_modules(image_encoder=encoder, feature_extractor=processor)
        projection = IPAdapterPlusImageProjection(embed_dims=32, output_dims=768, hidden_dims=32,
            depth=1, dim_head=8, heads=4, num_queries=4)
        pipe.unet.encoder_hid_proj = MultiIPAdapterImageProjection([projection])
        pipe.unet.register_to_config(encoder_hid_dim_type="ip_image_proj")
        processors = {}
        for name, current in pipe.unet.attn_processors.items():
            if name.endswith("attn1.processor") or name.startswith("encoder_hid_proj"):
                processors[name] = current
                continue
            if name.startswith("mid_block"):
                channels = 64
            elif name.startswith("up_blocks"):
                channels = (64, 32)[int(name.split(".")[1])]
            else:
                channels = (32, 64)[int(name.split(".")[1])]
            processors[name] = IPAdapterAttnProcessor2_0(channels, cross_attention_dim=768, num_tokens=(4,))
        pipe.unet.set_attn_processor(processors)
        cfg["use_ip_adapter"] = True
        conditioned = AdvancedInpaintPass(pipe, cfg, canvas, "cpu", 42)
        with torch.inference_mode():
            output = conditioned(image=canvas, mask_image=Image.fromarray(mask),
                prompt_embeds=torch.zeros((1, 4, 768)), negative_prompt_embeds=torch.zeros((1, 4, 768)),
                height=32, width=32, strength=1., num_inference_steps=2, guidance_scale=7.,
                generator=torch.Generator().manual_seed(42), output_type="latent")
        self.assertEqual(tuple(output.images.shape), (1, 4, 16, 16))
        self.assertTrue(conditioned.diagnostics["use_ip_adapter"])
        self.assertEqual(len(conditioned.diagnostics["latent_steps"]), 2)

    def test_ddim_callback_at_full_and_partial_strength(self):
        import torch
        from diffusers import DDIMScheduler
        from src.models.sd15_ip_adapter import ReferenceLatentBlender

        for strength in (1., .24, .62):
            scheduler = DDIMScheduler(num_train_timesteps=1000, set_alpha_to_one=True,
                                      steps_offset=0, timestep_spacing="leading", clip_sample=False)
            scheduler.set_timesteps(50)
            timesteps = scheduler.timesteps[50 - int(50 * strength):]
            clean = torch.full((1, 4, 8, 8), .3)
            noise = torch.ones_like(clean)
            editable = torch.zeros((1, 1, 8, 8))
            editable[:, :, 3:5, 3:5] = 1
            callback = ReferenceLatentBlender(clean, noise, editable)
            for index, timestep in enumerate(timesteps):
                # Actual DDIM.step followed by exactly the callback the pipeline invokes.
                current = scheduler.step(torch.zeros_like(clean), timestep, torch.ones_like(clean)).prev_sample
                result = callback(SimpleNamespace(scheduler=scheduler), index, timestep, {"latents": current.clone()})["latents"]
                torch.testing.assert_close(result[:, :, 3:5, 3:5], current[:, :, 3:5, 3:5])
                prev = int(timestep) - 20
                expected = scheduler.add_noise(clean, noise, torch.tensor([prev])) if prev >= 0 else clean
                torch.testing.assert_close(result[:, :, 0, 0], expected[:, :, 0, 0])
            torch.testing.assert_close(result[:, :, 0, 0], clean[:, :, 0, 0])
            self.assertEqual(len(callback.steps), len(timesteps))


if __name__ == "__main__":
    unittest.main()
