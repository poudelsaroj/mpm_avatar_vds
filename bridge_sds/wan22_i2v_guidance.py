"""
Wan2.2 TI2V-5B guidance via HuggingFace diffusers.

Uses WanPipeline (Wan-AI/Wan2.2-TI2V-5B-Diffusers) loaded from a local
snapshot_download directory.  No CPU offloading.  bfloat16 throughout.

Architecture (Wan2.2 TI2V-5B):
  - WanTransformer3DModel  — flow-prediction denoiser (bfloat16, frozen)
  - AutoencoderKLWan       — 4D VAE, spatial ×8 + temporal ×4 compression (float32)
  - T5EncoderModel         — umt5-xxl text encoder, 4096-dim embeddings (bfloat16, frozen)
  - UniPCMultistepScheduler — flow-matching, prediction_type='flow_prediction', T=1000

SDS objective (rectified-flow / flow-matching):
    σ = t / T  ∈ [0, 1]
    x_t = (1 − σ) · x₀ + σ · ε
    v_target = ε − x₀          (direction from data to noise)
    L_SDS = ‖v_θ(x_t, t, c) − v_target‖²

x₀ = VAE.encode(simulated_video) · scaling_factor
v_θ is the frozen Wan transformer.
SPSA probes the scalar loss with ±Δφ and estimates ∂L/∂φ numerically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Wan22I2VConfig:
    # Local path to a snapshot_download of Wan-AI/Wan2.2-TI2V-5B-Diffusers
    ckpt_dir: Path

    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16  # for transformer; VAE always float32

    # Text conditioning (empty → unconditional baseline works fine for SDS)
    prompt: str = ""
    negative_prompt: str = ""
    use_cfg: bool = False
    cfg_scale: float = 3.5

    # Flow-matching schedule length — must match model (Wan uses 1000)
    num_train_timesteps: int = 1000

    # ── Legacy fields kept for backwards-compat with train_sds_physics.py ──────
    # These are accepted but unused in the diffusers-based implementation.
    wan_repo_root: Optional[Path] = None
    boundary: float = 0.900
    use_first_frame_mask: bool = True
    t5_checkpoint_name: str = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer: str = "google/umt5-xxl"
    vae_checkpoint_name: str = "Wan2.1_VAE.pth"
    low_noise_subfolder: str = "low_noise_model"
    high_noise_subfolder: str = "high_noise_model"


class Wan22I2VGuidance(nn.Module):
    """
    Wraps Wan2.2-TI2V-5B (via HuggingFace diffusers) as a frozen SDS prior.

    All components live on device at all times — no CPU offloading.

    Components:
      - self.vae         : AutoencoderKLWan (float32)
      - self.transformer : WanTransformer3DModel (bfloat16, frozen)
      - self.text_encoder: T5EncoderModel / umt5-xxl (bfloat16, frozen)
      - self.tokenizer   : T5Tokenizer (max_length=512)
      - self.scheduler   : UniPCMultistepScheduler (flow_prediction)

    Public API:
      guidance.compute_loss(video_01, cond_image_01=None, timesteps=None, generator=None)
        → scalar SDS loss (float32 tensor)
    """

    def __init__(self, config: Wan22I2VConfig):
        super().__init__()
        self.cfg    = config
        self.device = torch.device(config.device)
        # Always bfloat16 for transformer (more stable than float16 for Wan)
        self._dtype = torch.bfloat16

        try:
            from diffusers import WanPipeline, AutoencoderKLWan  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "diffusers is required for Wan 5B guidance.\n"
                "Install: pip install git+https://github.com/huggingface/diffusers\n"
                f"Error: {exc}"
            ) from exc

        model_path = str(config.ckpt_dir)
        print(f"[Wan5B] Loading from {model_path} …")

        # VAE in float32 for better encoding quality
        vae = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.float32
        )
        # Full pipeline in bfloat16
        pipe = WanPipeline.from_pretrained(
            model_path, vae=vae, torch_dtype=self._dtype
        )
        pipe = pipe.to(self.device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        # Extract components and freeze
        self.vae: nn.Module          = pipe.vae.eval().requires_grad_(False)
        self.transformer: nn.Module  = pipe.transformer.eval().requires_grad_(False)
        self.text_encoder: nn.Module = pipe.text_encoder.eval().requires_grad_(False)
        self.tokenizer               = pipe.tokenizer
        self.scheduler               = pipe.scheduler

        # Cache scaling_factor — defaults to the Wan value 0.13235
        self._scaling_factor: float = float(
            getattr(getattr(self.vae, "config", None), "scaling_factor", None) or 0.13235
        )

        # Pre-encode text prompts (constant across training)
        with torch.no_grad():
            self._context      = self._encode_text(config.prompt)
            self._context_null = self._encode_text(config.negative_prompt or "")

        del pipe  # pipeline shell no longer needed; components stored in self.*
        print(
            f"[Wan5B] Ready — transformer=bfloat16, vae=float32, "
            f"scaling_factor={self._scaling_factor}, device={self.device}"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_train_timesteps(self) -> int:
        return int(getattr(self.scheduler.config, "num_train_timesteps", 1000))

    # ── Text encoding ─────────────────────────────────────────────────────────

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Tokenise + encode text → (1, seq_len, 4096) bfloat16.

        Passes attention_mask so padding tokens are not attended to.
        """
        max_len = int(getattr(self.tokenizer, "model_max_length", 512))
        tokens = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        input_ids      = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return out.last_hidden_state.to(self._dtype)   # (1, seq_len, 4096)

    # ── VAE helpers ───────────────────────────────────────────────────────────

    def _vae_encode(
        self,
        video_m11: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Encode video [-1, 1] → scaled latents.

        video_m11: (B, 3, T, H, W) float32 in [-1, 1]
        Returns:   (B, 16, T/4, H/8, W/8) float32, scaled by scaling_factor
        """
        posterior = self.vae.encode(video_m11)
        latent_dist = posterior.latent_dist

        # DiagonalGaussianDistribution.sample() signature varies across
        # diffusers versions — handle both with and without generator arg.
        try:
            x0 = latent_dist.sample(generator=generator)
        except TypeError:
            # Fallback: set manual seed on torch global RNG if generator given
            if generator is not None:
                torch.manual_seed(generator.initial_seed())
            x0 = latent_dist.sample()

        return x0 * self._scaling_factor

    # ── Transformer forward ───────────────────────────────────────────────────

    def _denoise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.LongTensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single frozen transformer forward pass.

        x_t       : (B, 16, T', H', W') bfloat16 — noisy latents
        timesteps : (B,) long
        context   : (B, seq_len, 4096) bfloat16 — text embeddings

        Returns:  (B, 16, T', H', W') float32 — predicted flow
        """
        # return_dict=False → returns tuple; [0] is the output tensor.
        # This avoids any potential Transformer2DModelOutput attribute issues.
        with torch.no_grad():
            out = self.transformer(
                hidden_states=x_t,
                timestep=timesteps,
                encoder_hidden_states=context,
                return_dict=False,
            )
        pred = out[0]  # (B, 16, T', H', W') bfloat16

        # Optional classifier-free guidance
        if self.cfg.use_cfg:
            context_null = self._context_null.expand(x_t.shape[0], -1, -1)
            with torch.no_grad():
                out_uncond = self.transformer(
                    hidden_states=x_t,
                    timestep=timesteps,
                    encoder_hidden_states=context_null,
                    return_dict=False,
                )
            pred_uncond = out_uncond[0]
            pred = pred_uncond + float(self.cfg.cfg_scale) * (pred - pred_uncond)

        return pred.float()

    # ── SDS loss ──────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        video_01: torch.Tensor,
        cond_image_01: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        One-step rectified-flow SDS loss.

        Args:
            video_01      : (B, 3, T, H, W) float32 in [0, 1] — simulated video
            cond_image_01 : (3, H, W) or (B, 3, H, W) — optional first-frame
                            conditioning image (accepted for API compat; the 5B
                            model is used in T2V mode so this is currently unused)
            timesteps     : (B,) long in [0, num_train_timesteps).
                            Sampled uniformly at random if None.
            generator     : torch.Generator for reproducible noise across SPSA
                            ± perturbations.  Pass the same generator (reset to
                            the same seed) for every probe in one SPSA step.

        Returns:
            Scalar SDS loss tensor (float32).
            For SPSA, detach() is not needed — the caller wraps in no_grad().
        """
        if video_01.ndim != 5 or video_01.shape[1] != 3:
            raise ValueError(
                f"video_01 must be (B, 3, T, H, W), got shape {tuple(video_01.shape)}"
            )
        b, _, T_frames, H, W = video_01.shape
        # Wan VAE compresses temporal ×4 — need at least 4 frames for 1 latent frame
        if T_frames < 4:
            raise ValueError(
                f"video_01 has only {T_frames} frames. "
                "Wan VAE requires ≥4 frames (temporal compression ×4). "
                "Set TRAIN_FRAME_NUM ≥ 16 in your config."
            )

        # ── 1. Encode video to scaled latents ────────────────────────────────
        video_m11 = (video_01 * 2.0 - 1.0).to(device=self.device, dtype=torch.float32)  # [-1, 1]
        x0 = self._vae_encode(video_m11, generator=generator)
        # x0: (B, 16, T/4, H/8, W/8) float32

        # ── 2. Sample diffusion timestep ─────────────────────────────────────
        if timesteps is None:
            timesteps = torch.randint(
                0, self.num_train_timesteps, (b,),
                generator=generator, device=self.device,
            ).long()
        timesteps = timesteps.to(self.device)

        # ── 3. Rectified-flow noising ─────────────────────────────────────────
        #   x_t = (1 − σ) x₀ + σ ε,    σ = t / T ∈ [0, 1]
        noise = torch.randn_like(x0, generator=generator)
        sigma = (timesteps.float() / float(self.num_train_timesteps))
        # Reshape for broadcasting: (B,) → (B, 1, 1, 1, 1)
        sigma = sigma.view(b, 1, 1, 1, 1)
        x_t = (1.0 - sigma) * x0 + sigma * noise

        # Target flow: model should predict direction from x₀ to noise
        target = noise - x0  # (B, 16, T', H', W') float32

        # ── 4. Frozen denoiser forward ────────────────────────────────────────
        context = self._context.expand(b, -1, -1)  # (B, 512, 4096) bfloat16
        pred = self._denoise(x_t.to(self._dtype), timesteps, context)
        # pred: (B, 16, T', H', W') float32

        # ── 5. MSE flow loss ──────────────────────────────────────────────────
        return F.mse_loss(pred, target, reduction="mean")
