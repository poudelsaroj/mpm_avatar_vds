"""
bridge_sds/scorer_phase3.py
============================
Score Distillation Sampling (SDS) scorer using Dreamcloth Phase3 / Wan I2V.

STATUS: STUB — Phase3 SDS scoring API wired but NOT yet active.
        The runner, optimiser, and regularisers are fully functional.
        This file will be completed once the Wan I2V checkpoint path and
        Phase3 import interface are confirmed on Clariden / Bristen.

CRITICAL CONSTRAINT
-------------------
This module will REFUSE to run silently with random weights.
If `ckpt_path` is provided but the file does not exist, or if no checkpoint
is provided and `sds.enabled = true`, it raises a hard error explaining that
SDS requires a trained denoiser.

To run the optimiser without SDS (physical regularisers only):
    --sds_enabled false   (or set sds.enabled: false in the YAML config)

Wan I2V integration notes (for Clariden/Bristen)
-------------------------------------------------
- Wan is already running on Clariden; expose it as a scoring endpoint.
- The scoring API below accepts [T, H, W, 3] float32 in [0, 1].
- Frames are resized to `target_resolution` (default 128×128) before scoring.
- FP16 AMP is used automatically if `use_amp = True`.
- Diffusion timestep is uniformly sampled from [timestep_min, timestep_max].

SDS gradient formula (for reference)
--------------------------------------
    ∇_φ L_SDS  ≈  E_t,ε [ w(t) · (ε_θ(z_t, t) − ε) · ∂z / ∂φ ]

where z = encoder(rendered_frames), ε_θ is the denoiser, ε ~ N(0,I).
In black-box / SPSA mode the Jacobian ∂z/∂φ is NOT needed; only the scalar
loss value J = ||ε_θ(z_t, t) − ε||² is used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .utils_video_io import resize_frames

logger = logging.getLogger(__name__)

class Phase3Scorer:
    """
    Wraps the Dreamcloth Phase3 / Wan I2V video diffusion model as a
    scalar SDS loss function.

    Args:
        ckpt_path        : path to the pretrained Wan I2V checkpoint.
                           REQUIRED when sds.enabled=True.
                           Pass None only with sds.enabled=False.
        device           : torch device string
        target_resolution: (H, W) to resize frames before scoring
        timestep_range   : (min, max) DDPM timestep range
        guidance_scale   : classifier-free guidance weight
        use_amp          : use FP16 automatic mixed precision
    """

    def __init__(
        self,
        ckpt_path:         Optional[Union[str, Path]],
        device:            str = "cuda:0",
        target_resolution: int = 128,
        timestep_range:    Tuple[int, int] = (50, 950),
        guidance_scale:    float = 7.5,
        use_amp:           bool = True,
    ):
        self.device           = device
        self.target_res       = target_resolution
        self.timestep_range   = timestep_range
        self.guidance_scale   = guidance_scale
        self.use_amp          = use_amp
        self._model           = None
        self._model_ready     = False

        # ── Checkpoint validation ────────────────────────────────────────────
        if ckpt_path is None:
            raise ValueError(
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  Phase3Scorer: NO CHECKPOINT PROVIDED                   ║\n"
                "║                                                          ║\n"
                "║  SDS requires a trained Wan I2V denoiser checkpoint.    ║\n"
                "║  Pass --phase3_ckpt /path/to/wan_i2v.ckpt               ║\n"
                "║  OR disable SDS with --no_sds                           ║\n"
                "╚══════════════════════════════════════════════════════════╝"
            )

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  Phase3Scorer: CHECKPOINT NOT FOUND                     ║\n"
                f"║  Path: {str(ckpt_path):<52}║\n"
                "║                                                          ║\n"
                "║  SDS needs a trained Wan I2V denoiser.                  ║\n"
                "║  Options:                                                ║\n"
                "║    1) Provide the correct path with --phase3_ckpt       ║\n"
                "║    2) Disable SDS:  --sds_enabled false                 ║\n"
                "║       (optimiser will use only physical regularisers)   ║\n"
                "╚══════════════════════════════════════════════════════════╝"
            )

        self._ckpt_path = ckpt_path

        # ── Load model ───────────────────────────────────────────────────────
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        """
        Load the Wan I2V / Phase3 video diffusion model.

        TODO (Clariden/Bristen integration):
        ─────────────────────────────────────
        Replace the placeholder below with the actual Wan model loader.
        The expected interface from Dreamcloth Phase3:

            sys.path.insert(0, str(DREAMCLOTH_ROOT / "phase3"))
            from video_diffusion_model import VideoDiffusionModel
            self._model = VideoDiffusionModel(...)
            ckpt = torch.load(self._ckpt_path, map_location="cpu")
            self._model.load_state_dict(ckpt["model_state_dict"])
            self._model.eval().to(self.device)
            for p in self._model.parameters():
                p.requires_grad_(False)

        For Wan 2.1/2.2 I2V (already deployed on Clariden), the interface is:

            from wan.modules.model import WanModel
            from wan.configs import WAN_CONFIGS
            cfg = WAN_CONFIGS["wan2.1-i2v-14B"]
            self._model = WanModel.from_pretrained(self._ckpt_path, cfg)
            self._model.eval().to(self.device)
        """
        logger.warning(
            "Phase3Scorer._load_model(): STUB — "
            "Wan I2V integration not yet wired. "
            "score() will raise until model loading is implemented. "
            "Complete the TODO block in scorer_phase3.py on Clariden."
        )
        # Placeholder: store path, flag as NOT ready
        self._model       = None
        self._model_ready = False

    # ── Scoring API ───────────────────────────────────────────────────────────

    def score(
        self,
        frames:  torch.Tensor,
        seed:    Optional[int] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute the SDS loss for a video clip.

        Args:
            frames           : [T, H, W, 3] float32 in [0, 1]
            seed             : random seed for reproducible noise sampling
                               (use the same seed for both SPSA ± evaluations)
            return_components: if True, also return a dict of sub-losses

        Returns:
            loss_sds            : scalar tensor (detached from Phase3 graph)
            components (opt)    : {'sds': float, 'grad_norm': float, ...}
        """
        if not self._model_ready or self._model is None:
            raise RuntimeError(
                "Phase3Scorer is not ready. "
                "A checkpoint path was provided but model loading is still stubbed. "
                "Complete bridge_sds/scorer_phase3.py::_load_model() for Wan/Phase3 "
                "or run with --no_sds."
            )

        # ── Real SDS path (active once Wan is wired) ─────────────────────────
        return self._compute_sds(frames, seed=seed, return_components=return_components)

    @torch.no_grad()
    def _compute_sds(
        self,
        frames: torch.Tensor,
        seed:   Optional[int] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Actual SDS computation.  Called only when self._model is loaded.

        Algorithm (DDPM-SDS, Poole et al. 2022):
          1. Resize frames to target_resolution and encode to latents z.
          2. Sample random timestep t ∈ [t_min, t_max].
          3. Sample noise ε ~ N(0, I).
          4. Noise the latents:  z_t = sqrt(ᾱ_t) z + sqrt(1-ᾱ_t) ε
          5. Denoiser prediction:  ε_θ = model(z_t, t, conditioning)
          6. SDS loss:  L = w(t) * ||ε_θ - ε||²
        """
        device = self.device

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        # ── 1. Preprocess frames ─────────────────────────────────────────────
        T, H, W, C = frames.shape
        if H != self.target_res or W != self.target_res:
            frames = resize_frames(frames, self.target_res, self.target_res)

        # [T, H, W, 3] → [1, C, T, H, W]  (batch=1, channels-first video tensor)
        video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # [1, 3, T, H, W]
        video = video * 2.0 - 1.0  # [0,1] → [-1,1]

        # ── 2. Encode to latents ──────────────────────────────────────────────
        # TODO: call self._model.encode(video) or VAE encoder
        # z = self._model.vae.encode(video).latent_dist.sample()
        # For now: use video directly as "latent" (no VAE)
        z = video

        # ── 3. Sample timestep ────────────────────────────────────────────────
        t_min, t_max = self.timestep_range
        t = torch.randint(t_min, t_max + 1, (1,), device=device, generator=generator)

        # ── 4. Get noise schedule coefficients ───────────────────────────────
        # TODO: get alphas_cumprod from model's noise scheduler
        # alpha_bar_t = self._model.scheduler.alphas_cumprod[t]
        # For stub: use cosine schedule approximation
        alpha_bar_t = torch.cos(t.float() / 1000 * (torch.pi / 2)).pow(2)

        # ── 5. Add noise ──────────────────────────────────────────────────────
        eps = torch.randn_like(z, generator=generator)
        z_t = alpha_bar_t.sqrt() * z + (1 - alpha_bar_t).sqrt() * eps

        # ── 6. Denoiser prediction ────────────────────────────────────────────
        autocast_ctx = torch.cuda.amp.autocast() if self.use_amp else torch.no_grad()
        with autocast_ctx:
            # TODO: replace with actual Wan forward call
            # eps_theta = self._model(z_t, t, conditioning=None)
            eps_theta = torch.zeros_like(z_t)  # placeholder

        # ── 7. SDS loss ───────────────────────────────────────────────────────
        w_t = (1 - alpha_bar_t).sqrt()               # SNR weighting
        sds_grad = w_t * (eps_theta - eps)
        loss_sds  = (sds_grad.detach() * z).sum()    # stop-gradient on denoiser side

        grad_norm = sds_grad.norm().item()

        if return_components:
            return loss_sds, {
                "sds":       float(loss_sds.item()),
                "grad_norm": grad_norm,
                "timestep":  int(t.item()),
            }
        return loss_sds

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the model is loaded and ready for scoring."""
        return self._model_ready and self._model is not None

    def __repr__(self) -> str:
        ckpt = str(self._ckpt_path) if self._ckpt_path else "None"
        return (
            f"Phase3Scorer("
            f"ready={self.is_ready}, "
            f"ckpt={ckpt}, "
            f"device={self.device}, "
            f"target_res={self.target_res})"
        )
