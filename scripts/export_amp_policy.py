"""Export an AMP-style actor checkpoint as a self-contained TorchScript model.

The training-side checkpoint stores actor MLP weights alongside running stats
for input and action normalizers. This script bakes the MLP, both
normalizers, and the action-clip into a single scripted module so the deploy
side can just `torch.jit.load(...)` and run forward — without knowing the
network shape, normalizer keys, or dimensions.

Expected state-dict keys (auto-detected from the checkpoint):
    _obs_norm._mean, _obs_norm._std           - observation normalizer
    _a_norm._mean,   _a_norm._std             - action normalizer
    _model._actor_layers.{0,2,4,...}.{weight,bias}  - actor MLP body (Linear+ReLU stack)
    _model._action_dist._mean_net.{weight,bias}     - action-mean head

Hidden sizes are inferred from the actor weight shapes — no flags needed.

Forward: obs_raw[obs_dim] -> action_clipped[action_dim]
    norm     = clamp((obs - obs_mean) / obs_std, -clip, clip)
    features = MLP(norm)
    action   = features * a_std + a_mean
    return clamp(action, a_mean - a_std, a_mean + a_std)

EMA action smoothing is intentionally left out — it's stateful and lives on
the deploy side.

Usage:
    python scripts/export_amp_policy.py \
        --ckpt path/to/state_dict.pt \
        --out  path/to/scripted_model.pt
"""
from __future__ import annotations

import argparse
import re

import torch
import torch.nn as nn

NORMALIZED_OBSERVATION_CLIP = 10.0
MIN_NORMALIZER_STD = 1e-5


def _infer_actor_layer_indices(state_dict: dict[str, torch.Tensor]) -> list[int]:
    pattern = re.compile(r"^_model\._actor_layers\.(\d+)\.weight$")
    indices = sorted(
        int(m.group(1)) for m in (pattern.match(k) for k in state_dict) if m
    )
    if not indices:
        raise ValueError(
            "No `_model._actor_layers.N.weight` keys found in checkpoint."
        )
    return indices


class ScriptedActorPolicy(nn.Module):
    """obs_raw -> normalize -> MLP -> denormalize -> clip."""

    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        super().__init__()

        obs_mean = state_dict["_obs_norm._mean"].to(torch.float32)
        obs_std = torch.clamp(
            state_dict["_obs_norm._std"].to(torch.float32), min=MIN_NORMALIZER_STD
        )
        a_mean = state_dict["_a_norm._mean"].to(torch.float32)
        a_std = state_dict["_a_norm._std"].to(torch.float32)

        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", obs_std)
        self.register_buffer("a_mean", a_mean)
        self.register_buffer("a_std", a_std)
        self.register_buffer(
            "action_lower", a_mean - a_std, persistent=False
        )
        self.register_buffer(
            "action_upper", a_mean + a_std, persistent=False
        )

        actor_indices = _infer_actor_layer_indices(state_dict)
        layers: list[nn.Module] = []
        for i, idx in enumerate(actor_indices):
            w = state_dict[f"_model._actor_layers.{idx}.weight"]
            b = state_dict[f"_model._actor_layers.{idx}.bias"]
            linear = nn.Linear(w.shape[1], w.shape[0])
            linear.weight.data = w
            linear.bias.data = b
            layers.append(linear)
            layers.append(nn.ReLU())

        head_w = state_dict["_model._action_dist._mean_net.weight"]
        head_b = state_dict["_model._action_dist._mean_net.bias"]
        head = nn.Linear(head_w.shape[1], head_w.shape[0])
        head.weight.data = head_w
        head.bias.data = head_b
        layers.append(head)

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm = torch.clamp((obs - self.obs_mean) / self.obs_std, -10.0, 10.0)
        normalized_action = self.mlp(norm.unsqueeze(0)).squeeze(0)
        action = normalized_action * self.a_std + self.a_mean
        return torch.clamp(action, self.action_lower, self.action_upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Training-side state_dict .pt")
    parser.add_argument("--out", required=True, help="Output TorchScript .pt")
    args = parser.parse_args()

    state_dict = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    module = ScriptedActorPolicy(state_dict).eval()

    scripted = torch.jit.script(module)

    with torch.no_grad():
        obs = torch.randn(module.obs_mean.shape[0], dtype=torch.float32)
        eager_action = module(obs)
        scripted_action = scripted(obs)
    max_diff = (eager_action - scripted_action).abs().max().item()
    assert max_diff < 1e-6, f"scripted vs eager mismatch: {max_diff}"

    scripted.save(args.out)
    obs_dim = module.obs_mean.shape[0]
    act_dim = module.a_mean.shape[0]
    hidden = [
        layer.out_features for layer in module.mlp if isinstance(layer, nn.Linear)
    ][:-1]
    print(
        f"wrote {args.out}\n"
        f"  obs_dim={obs_dim}  act_dim={act_dim}  hidden={hidden}\n"
        f"  sanity max_diff={max_diff:.2e}"
    )


if __name__ == "__main__":
    main()
