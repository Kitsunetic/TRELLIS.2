from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from trellis2.pipelines.samplers.classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler
from trellis2.pipelines.samplers.guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerGeometryGuidanceSampler(
    GuidanceIntervalSamplerMixin,
    ClassifierFreeGuidanceSamplerMixin,
    FlowEulerSampler,
):
    """
    Euler sampler with existing sparse CFG and an additional geometry-guidance
    gradient on predicted x0.
    """

    def _geometry_guidance_active(self, t: float, interval: Tuple[float, float]) -> bool:
        return interval[0] <= t <= interval[1]

    def _geometry_guidance_scale(
        self,
        t: float,
        strength: float,
        interval: Tuple[float, float],
        schedule: str,
    ) -> float:
        if schedule == "constant":
            return strength

        denom = max(interval[1] - interval[0], 1e-6)
        normalized_t = (t - interval[0]) / denom
        normalized_t = float(np.clip(normalized_t, 0.0, 1.0))

        if schedule == "linear_decay":
            return strength * normalized_t
        if schedule == "linear_rise":
            return strength * (1.0 - normalized_t)

        raise ValueError(f"Unsupported geometry guidance schedule: {schedule}")

    def _clip_grad(self, grad: torch.Tensor, grad_clip: Optional[float]) -> torch.Tensor:
        if grad_clip is None:
            return grad
        flat = grad.reshape(grad.shape[0], -1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        factors = torch.clamp(grad_clip / norms, max=1.0)
        return grad * factors.view(-1, *([1] * (grad.ndim - 1)))

    def _sample_once_with_geometry_guidance(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond,
        *,
        neg_cond,
        guidance_strength: float,
        guidance_interval: Tuple[float, float],
        guidance_rescale: float,
        geometry_guidance,
        geometry_guidance_strength: float,
        geometry_guidance_interval: Tuple[float, float],
        geometry_guidance_schedule: str,
        geometry_guidance_rescale: bool,
        geometry_grad_clip: Optional[float],
        geometry_guidance_cfg_mode: str,
        **kwargs,
    ):
        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_(True)
            if geometry_guidance_cfg_mode == "with_cfg":
                pred_x_0, _, _ = self._get_model_prediction(
                    model,
                    x_t,
                    t,
                    cond,
                    neg_cond=neg_cond,
                    guidance_strength=guidance_strength,
                    guidance_interval=guidance_interval,
                    guidance_rescale=guidance_rescale,
                    **kwargs,
                )
            elif geometry_guidance_cfg_mode == "cond_only":
                pred_v_guidance = FlowEulerSampler._inference_model(self, model, x_t, t, cond, **kwargs)
                pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v_guidance)
            else:
                raise ValueError(f"Unsupported geometry guidance CFG mode: {geometry_guidance_cfg_mode}")

            loss, metrics = geometry_guidance.compute_loss(pred_x_0, t)
            grad = torch.autograd.grad(loss, x_t, only_inputs=True)[0]
            grad = self._clip_grad(grad, geometry_grad_clip)

        with torch.no_grad():
            pred_x_0_sample, _, pred_v = self._get_model_prediction(
                model,
                x_t.detach(),
                t,
                cond,
                neg_cond=neg_cond,
                guidance_strength=guidance_strength,
                guidance_interval=guidance_interval,
                guidance_rescale=guidance_rescale,
                **kwargs,
            )

            if geometry_guidance_rescale:
                grad_rms = grad.square().mean().sqrt().clamp_min(1e-8)
                pred_rms = pred_v.detach().square().mean().sqrt().clamp_min(1e-8)
                grad = grad * (pred_rms / grad_rms)

            scale = self._geometry_guidance_scale(
                t,
                geometry_guidance_strength,
                geometry_guidance_interval,
                geometry_guidance_schedule,
            )
            pred_v = pred_v + scale * grad
            pred_x_prev = x_t.detach() - (t - t_prev) * pred_v

        metrics = dict(metrics)
        metrics["geometry_grad_rms"] = float(grad.detach().square().mean().sqrt().item())
        metrics["geometry_guidance_scale"] = float(scale)
        return edict({"pred_x_prev": pred_x_prev.detach(), "pred_x_0": pred_x_0_sample.detach(), "guidance": metrics})

    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        guidance_rescale: float = 0.0,
        geometry_guidance=None,
        geometry_guidance_strength: float = 1.0,
        geometry_guidance_interval: Tuple[float, float] = (0.5, 0.95),
        geometry_guidance_schedule: str = "constant",
        geometry_guidance_rescale: bool = True,
        geometry_grad_clip: Optional[float] = 5.0,
        geometry_guidance_cfg_mode: str = "cond_only",
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs,
    ):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()

        control_latent = kwargs.pop("control", None)
        tau_0 = kwargs.pop("space_control_tau", 6)
        start_step_idx = 0
        if control_latent is not None:
            start_step_idx = min(max(tau_0, 0), steps - 1)
            t_0 = t_seq[start_step_idx]
            sample = t_0 * noise + (1.0 - t_0) * control_latent

        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(start_step_idx, steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": [], "guidance": []})

        base_args: Dict[str, Any] = {
            "neg_cond": neg_cond,
            "guidance_strength": guidance_strength,
            "guidance_interval": guidance_interval,
            "guidance_rescale": guidance_rescale,
        }
        base_args.update(kwargs)

        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            geometry_active = geometry_guidance is not None and self._geometry_guidance_active(t, geometry_guidance_interval)
            if geometry_active:
                out = self._sample_once_with_geometry_guidance(
                    model,
                    sample,
                    t,
                    t_prev,
                    cond,
                    geometry_guidance=geometry_guidance,
                    geometry_guidance_strength=geometry_guidance_strength,
                    geometry_guidance_interval=geometry_guidance_interval,
                    geometry_guidance_schedule=geometry_guidance_schedule,
                    geometry_guidance_rescale=geometry_guidance_rescale,
                    geometry_grad_clip=geometry_grad_clip,
                    geometry_guidance_cfg_mode=geometry_guidance_cfg_mode,
                    **base_args,
                )
                ret.guidance.append(out.guidance)
            else:
                with torch.no_grad():
                    out = super().sample_once(model, sample, t, t_prev, cond, **base_args)

            sample = out.pred_x_prev.detach()
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret
