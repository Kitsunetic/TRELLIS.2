from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from trellis2.pipelines.samplers.classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler
from trellis2.pipelines.samplers.guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerShapeGuidanceSampler(
    GuidanceIntervalSamplerMixin,
    ClassifierFreeGuidanceSamplerMixin,
    FlowEulerSampler,
):
    """
    Euler sampler for SparseTensor shape-SLat latents with an additional
    geometry-guidance gradient on x_t.feats.
    """

    def _guidance_active(self, t: float, interval: Tuple[float, float]) -> bool:
        return interval[0] <= t <= interval[1]

    def _guidance_scale(
        self,
        t: float,
        strength: float,
        interval: Tuple[float, float],
        schedule: str,
    ) -> float:
        if schedule == "constant":
            return strength

        denom = max(interval[1] - interval[0], 1e-6)
        normalized_t = float(np.clip((t - interval[0]) / denom, 0.0, 1.0))
        if schedule == "linear_decay":
            return strength * normalized_t
        if schedule == "linear_rise":
            return strength * (1.0 - normalized_t)
        raise ValueError(f"Unsupported shape guidance schedule: {schedule}")

    def _clip_sparse_grad(self, grad: torch.Tensor, grad_clip: Optional[float]) -> torch.Tensor:
        if grad_clip is None:
            return grad
        norm = grad.norm().clamp_min(1e-8)
        factor = torch.clamp(torch.as_tensor(grad_clip, device=grad.device, dtype=grad.dtype) / norm, max=1.0)
        return grad * factor

    def _sample_once_with_shape_guidance(
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
        shape_geometry_guidance,
        shape_geometry_guidance_strength: float,
        shape_geometry_guidance_interval: Tuple[float, float],
        shape_geometry_guidance_schedule: str,
        shape_geometry_guidance_rescale: bool,
        shape_geometry_grad_clip: Optional[float],
        shape_geometry_guidance_cfg_mode: str,
        shape_geometry_resolution: Optional[int],
        lambda_prior: float,
        **kwargs,
    ):
        with torch.enable_grad():
            x_feats = x_t.feats.detach().requires_grad_(True)
            x_t_grad = x_t.detach().replace(x_feats)
            if shape_geometry_guidance_cfg_mode == "with_cfg":
                pred_x0_guidance, _, _ = self._get_model_prediction(
                    model,
                    x_t_grad,
                    t,
                    cond,
                    neg_cond=neg_cond,
                    guidance_strength=guidance_strength,
                    guidance_interval=guidance_interval,
                    guidance_rescale=guidance_rescale,
                    **kwargs,
                )
            elif shape_geometry_guidance_cfg_mode == "cond_only":
                pred_v_guidance = FlowEulerSampler._inference_model(self, model, x_t_grad, t, cond, **kwargs)
                pred_x0_guidance, _ = self._v_to_xstart_eps(x_t=x_t_grad, t=t, v=pred_v_guidance)
            else:
                raise ValueError(f"Unsupported shape guidance CFG mode: {shape_geometry_guidance_cfg_mode}")

            loss, metrics = shape_geometry_guidance.compute_loss(pred_x0_guidance, t, resolution=shape_geometry_resolution)
            if lambda_prior > 0:
                with torch.no_grad():
                    _, _, pred_v_detached = self._get_model_prediction(
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
                    unguided_next = x_t.detach() - (t - t_prev) * pred_v_detached
                prior_loss = (x_t_grad.feats - unguided_next.feats).square().mean()
                loss = loss + lambda_prior * prior_loss
                metrics = dict(metrics)
                metrics["prior_loss"] = float(prior_loss.detach().item())

            grad = torch.autograd.grad(loss, x_feats, only_inputs=True)[0]
            grad = self._clip_sparse_grad(grad, shape_geometry_grad_clip)

        with torch.no_grad():
            pred_x0_sample, _, pred_v = self._get_model_prediction(
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

            if shape_geometry_guidance_rescale:
                grad_rms = grad.square().mean().sqrt().clamp_min(1e-8)
                pred_rms = pred_v.feats.detach().square().mean().sqrt().clamp_min(1e-8)
                grad = grad * (pred_rms / grad_rms)

            scale = self._guidance_scale(
                t,
                shape_geometry_guidance_strength,
                shape_geometry_guidance_interval,
                shape_geometry_guidance_schedule,
            )
            guided_v = pred_v.replace(pred_v.feats + scale * grad)
            pred_x_prev = x_t.detach() - (t - t_prev) * guided_v

        metrics = dict(metrics)
        metrics["shape_geometry_grad_rms"] = float(grad.detach().square().mean().sqrt().item())
        metrics["shape_geometry_guidance_scale"] = float(scale)
        return edict({"pred_x_prev": pred_x_prev.detach(), "pred_x_0": pred_x0_sample.detach(), "guidance": metrics})

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
        shape_geometry_guidance=None,
        shape_geometry_guidance_strength: float = 1.0,
        shape_geometry_guidance_interval: Tuple[float, float] = (0.5, 0.95),
        shape_geometry_guidance_schedule: str = "constant",
        shape_geometry_guidance_rescale: bool = True,
        shape_geometry_grad_clip: Optional[float] = 5.0,
        shape_geometry_guidance_cfg_mode: str = "cond_only",
        shape_geometry_resolution: Optional[int] = None,
        lambda_prior: float = 0.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs,
    ):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": [], "guidance": []})

        base_args: Dict[str, Any] = {
            "neg_cond": neg_cond,
            "guidance_strength": guidance_strength,
            "guidance_interval": guidance_interval,
            "guidance_rescale": guidance_rescale,
        }
        base_args.update(kwargs)

        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            active = shape_geometry_guidance is not None and self._guidance_active(t, shape_geometry_guidance_interval)
            if active:
                out = self._sample_once_with_shape_guidance(
                    model,
                    sample,
                    t,
                    t_prev,
                    cond,
                    shape_geometry_guidance=shape_geometry_guidance,
                    shape_geometry_guidance_strength=shape_geometry_guidance_strength,
                    shape_geometry_guidance_interval=shape_geometry_guidance_interval,
                    shape_geometry_guidance_schedule=shape_geometry_guidance_schedule,
                    shape_geometry_guidance_rescale=shape_geometry_guidance_rescale,
                    shape_geometry_grad_clip=shape_geometry_grad_clip,
                    shape_geometry_guidance_cfg_mode=shape_geometry_guidance_cfg_mode,
                    shape_geometry_resolution=shape_geometry_resolution,
                    lambda_prior=lambda_prior,
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
