"""
Wan2.2 Image-to-Video guidance wrapper for DreamCloth Phase3.

DreamCloth Phase3 needs a frozen "diffusion prior" that produces a training loss
whose gradients can flow to MPMParameters through a differentiable video path.

The official Wan2.2 I2V model is a *flow-prediction* diffusion model operating in
VAE latent space and conditioned on:
  - text embeddings (T5)
  - an image-derived conditional latent (first-frame conditioning + mask)

This wrapper exposes a single-step training objective:
  - encode simulated video x0 (latent)
  - sample timestep t and noise
  - construct noisy latent x_t via the Wan schedule
  - predict flow using Wan low/high-noise expert model
  - compute MSE between predicted flow and target flow

For Wan2.2 FlowUniPC schedule with prediction_type='flow_prediction' and
predict_x0=True, the model output v targets:
  v = (x_t - x0) / sigma  == noise - x0
given x_t = (1 - sigma) * x0 + sigma * noise.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Wan22I2VConfig:
    wan_repo_root: Optional[Path]
    ckpt_dir: Path

    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Conditioning
    prompt: str = ""
    negative_prompt: str = ""
    use_cfg: bool = False
    cfg_scale: float = 3.5

    # Expert switching
    boundary: float = 0.900  # same meaning as Wan configs: high-noise if t >= boundary * 1000

    # Schedule
    num_train_timesteps: int = 1000

    # Conditioning mask
    use_first_frame_mask: bool = True

    # Tokenizer/model locations
    t5_checkpoint_name: str = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer: str = "google/umt5-xxl"
    vae_checkpoint_name: str = "Wan2.1_VAE.pth"
    low_noise_subfolder: str = "low_noise_model"
    high_noise_subfolder: str = "high_noise_model"


def _normalize_video_to_m11(video_01: torch.Tensor) -> torch.Tensor:
    return video_01 * 2.0 - 1.0


def _image_to_tensor01(image: torch.Tensor) -> torch.Tensor:
    # Accept (3,H,W) or (B,3,H,W); ensure float in [0,1]
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    return image.clamp(0.0, 1.0)


class Wan22I2VGuidance(nn.Module):
    """
    Computes a Wan2.2 I2V flow-prediction loss from a pixel-space video tensor.

    This class assumes batch size 1 by default for feasibility with large models.
    """

    def __init__(self, config: Wan22I2VConfig):
        super().__init__()
        self.cfg = config
        self.device = torch.device(config.device)

        if config.wan_repo_root is not None:
            wan_root = str(config.wan_repo_root)
            if wan_root not in sys.path:
                sys.path.insert(0, wan_root)

        try:
            from wan.modules.model import WanModel  # type: ignore
            from wan.modules.t5 import T5EncoderModel  # type: ignore
            from wan.modules.vae2_1 import Wan2_1_VAE  # type: ignore
            from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import Wan2.2 modules. Ensure the Wan2.2 repo is installed or pass --wan-repo-root.\n"
                "Wan2.2 also requires diffusers/transformers dependencies."
            ) from e

        self._WanModel = WanModel
        self._T5EncoderModel = T5EncoderModel
        self._Wan2_1_VAE = Wan2_1_VAE
        self._Scheduler = FlowUniPCMultistepScheduler

        ckpt_dir = config.ckpt_dir
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Wan checkpoint dir not found: {ckpt_dir}")

        # VAE (frozen, but keep autograd for inputs)
        self.vae = self._Wan2_1_VAE(
            vae_pth=str(ckpt_dir / config.vae_checkpoint_name),
            dtype=config.dtype,
            device=str(self.device),
        )

        # Denoisers (frozen)
        self.low_noise_model = self._WanModel.from_pretrained(str(ckpt_dir), subfolder=config.low_noise_subfolder)
        self.high_noise_model = self._WanModel.from_pretrained(str(ckpt_dir), subfolder=config.high_noise_subfolder)
        self.low_noise_model.eval().requires_grad_(False).to(self.device)
        self.high_noise_model.eval().requires_grad_(False).to(self.device)

        # Scheduler (training schedule)
        self.scheduler = self._Scheduler(
            num_train_timesteps=config.num_train_timesteps,
            prediction_type="flow_prediction",
            shift=1.0,
            use_dynamic_shifting=False,
            predict_x0=True,
        )

        # Text encoder embeddings (precomputed; they do not depend on the simulation)
        t5_checkpoint_path = ckpt_dir / config.t5_checkpoint_name
        self.text_encoder = self._T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16 if config.dtype == torch.bfloat16 else torch.float16,
            device=torch.device("cpu"),
            checkpoint_path=str(t5_checkpoint_path),
            tokenizer_path=config.t5_tokenizer,
            shard_fn=None,
        )
        self.text_encoder.model.eval().requires_grad_(False)

        with torch.no_grad():
            # The Wan API returns a list[Tensor] (one per prompt in batch)
            self._context = self.text_encoder([config.prompt], torch.device("cpu"))[0].to(self.device)
            if config.negative_prompt:
                self._context_null = self.text_encoder([config.negative_prompt], torch.device("cpu"))[0].to(self.device)
            else:
                self._context_null = self.text_encoder([""], torch.device("cpu"))[0].to(self.device)

    @property
    def num_train_timesteps(self) -> int:
        return int(self.cfg.num_train_timesteps)

    def _build_mask(self, latent_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        # latent shape: (C, T, H, W); mask is 4 channels as used in Wan2.2 I2V y-conditioning.
        _, t, h, w = latent_shape
        mask = torch.zeros(4, t, h, w, device=self.device, dtype=torch.float32)
        mask[:, 0] = 1.0
        return mask

    @torch.no_grad()
    def _encode_cond_latent(self, cond_image_01: torch.Tensor, video_hw: Tuple[int, int], video_len: int) -> torch.Tensor:
        # Build a conditional video with the image as the first frame and zeros afterwards.
        h, w = video_hw
        img = _image_to_tensor01(cond_image_01).to(device=self.device, dtype=torch.float32)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
        img_m11 = _normalize_video_to_m11(img)

        video = torch.zeros(1, 3, video_len, h, w, device=self.device, dtype=torch.float32)
        video[:, :, 0] = img_m11[:, :, 0]
        video = video.squeeze(0)  # (3, T, H, W)

        y_lat = self.vae.encode([video])[0]  # (C_lat, T', H', W')
        mask = self._build_mask(tuple(y_lat.shape))
        return torch.cat([mask, y_lat], dim=0)

    def _choose_model(self, t: torch.Tensor) -> nn.Module:
        threshold = self.cfg.boundary * float(self.cfg.num_train_timesteps)
        return self.high_noise_model if float(t.item()) >= threshold else self.low_noise_model

    def compute_loss(
        self,
        video_01: torch.Tensor,
        cond_image_01: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Args:
            video_01: (B, 3, T, H, W) in [0, 1]
            cond_image_01: (3, H, W) or (B, 3, H, W) in [0, 1]
            timesteps: optional (B,) timesteps in [0, num_train_timesteps)
        """
        if video_01.ndim != 5 or video_01.shape[1] != 3:
            raise ValueError("video_01 must be (B, 3, T, H, W)")
        b, _, t_frames, h, w = video_01.shape
        if b != 1:
            raise ValueError("Wan22I2VGuidance currently supports batch_size=1 (due to model size and per-step expert selection).")

        if timesteps is None:
            timesteps = torch.randint(
                0, self.cfg.num_train_timesteps, (b,),
                generator=generator, device=self.device,
            ).long()
        t_scalar = timesteps[0].float()

        # Encode x0 with gradients (depends on simulated video).
        video_m11 = _normalize_video_to_m11(video_01.to(device=self.device, dtype=torch.float32))
        x0 = self.vae.encode([video_m11[0]])[0]  # (C_lat, T', H', W')

        # Conditioning y is constant wrt simulation, so compute without grads.
        y = self._encode_cond_latent(cond_image_01, (h, w), t_frames)

        # Noise schedule: sigma in [0, 1]. Use a simple mapping consistent with Wan timesteps.
        sigma = (t_scalar / float(self.cfg.num_train_timesteps)).clamp(0.0, 1.0)
        noise = torch.randn_like(x0, generator=generator)
        x_t = (1.0 - sigma) * x0 + sigma * noise

        # Target flow for flow_prediction + predict_x0: v = (x_t - x0)/sigma = noise - x0
        target_flow = noise - x0

        model = self._choose_model(t_scalar)

        # WanModel.forward expects lists.
        x_list = [x_t]
        y_list = [y]
        context_list = [self._context]
        seq_len = int((x0.shape[1] * (x0.shape[2] // 2) * (x0.shape[3] // 2)))
        t_tensor = t_scalar.view(1).to(device=self.device, dtype=torch.float32)

        pred_cond = model(x_list, t=t_tensor, context=context_list, seq_len=seq_len, y=y_list)[0]

        if self.cfg.use_cfg:
            pred_uncond = model(
                x_list,
                t=t_tensor,
                context=[self._context_null],
                seq_len=seq_len,
                y=y_list,
            )[0]
            pred = pred_uncond + float(self.cfg.cfg_scale) * (pred_cond - pred_uncond)
        else:
            pred = pred_cond

        return F.mse_loss(pred, target_flow, reduction="mean")

