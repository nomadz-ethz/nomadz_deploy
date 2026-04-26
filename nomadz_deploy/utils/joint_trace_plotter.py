from __future__ import annotations

import atexit
import math
import os
from html import escape
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def resolve_policy_trace_stem() -> str | None:
    """Return the configured policy-trace output stem, if any."""
    trace_stem = os.environ.get("BOOSTER_POLICY_LOG_PATH")
    if trace_stem:
        return trace_stem

    legacy_trace_stem = os.environ.get("MIMICKIT_STEERING_LOG_PATH")
    if legacy_trace_stem:
        if legacy_trace_stem.endswith("_policy"):
            return legacy_trace_stem
        return f"{legacy_trace_stem}_policy"

    return None


class JointTracePlotHelper:
    """Record actual/target joint traces and write NPZ + SVG outputs."""

    def __init__(
        self,
        output_stem: str | None,
        joint_names: Sequence[str],
        title: str,
    ) -> None:
        self.output_stem = output_stem
        self.joint_names = list(joint_names)
        self.title = title
        self._time_s: list[float] = []
        self._actual_joint_pos: list[np.ndarray] = []
        self._target_joint_pos: list[np.ndarray] = []
        self._last_flush_sample_count = 0

        if self.output_stem is not None:
            atexit.register(self.flush)

    @property
    def enabled(self) -> bool:
        return self.output_stem is not None

    def reset(self) -> None:
        self._time_s.clear()
        self._actual_joint_pos.clear()
        self._target_joint_pos.clear()
        self._last_flush_sample_count = 0

    def record(
        self,
        time_s: float,
        actual_joint_pos: torch.Tensor | np.ndarray,
        target_joint_pos: torch.Tensor | np.ndarray,
    ) -> None:
        if not self.enabled:
            return

        actual_np = self._to_numpy(actual_joint_pos)
        target_np = self._to_numpy(target_joint_pos)

        if actual_np.shape != target_np.shape:
            raise ValueError(
                "actual_joint_pos and target_joint_pos must have identical shapes, "
                f"got {actual_np.shape} and {target_np.shape}"
            )
        if actual_np.shape[0] != len(self.joint_names):
            raise ValueError(
                "joint trace shape must match joint_names length, "
                f"got {actual_np.shape[0]} values for {len(self.joint_names)} joints"
            )

        self._time_s.append(float(time_s))
        self._actual_joint_pos.append(actual_np)
        self._target_joint_pos.append(target_np)

    def flush(self) -> None:
        if not self.enabled:
            return

        num_samples = len(self._time_s)
        if num_samples == 0 or num_samples == self._last_flush_sample_count:
            return

        try:
            output_stem = Path(self.output_stem)
            output_stem.parent.mkdir(parents=True, exist_ok=True)

            time_s = np.asarray(self._time_s, dtype=np.float32)
            actual_joint_pos = np.stack(self._actual_joint_pos).astype(np.float32)
            target_joint_pos = np.stack(self._target_joint_pos).astype(np.float32)

            np.savez(
                f"{self.output_stem}.npz",
                time_s=time_s,
                actual_joint_pos=actual_joint_pos,
                target_joint_pos=target_joint_pos,
                joint_names=np.asarray(self.joint_names),
            )
            Path(f"{self.output_stem}.svg").write_text(
                self._build_svg(time_s, actual_joint_pos, target_joint_pos),
                encoding="utf-8",
            )
            self._last_flush_sample_count = num_samples
            print(
                f"saved {self.output_stem}.npz and {self.output_stem}.svg "
                f"with {num_samples} control steps"
            )
        except Exception as exc:
            print(f"failed to save policy joint trace {self.output_stem}: {exc}")

    @staticmethod
    def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=np.float32).reshape(-1).copy()

    @staticmethod
    def _sample_indices(num_points: int, max_points: int = 400) -> np.ndarray:
        if num_points <= max_points:
            return np.arange(num_points, dtype=np.int32)
        indices = np.linspace(0, num_points - 1, num=max_points, dtype=np.int32)
        indices[-1] = num_points - 1
        return np.unique(indices)

    def _build_svg(
        self,
        time_s: np.ndarray,
        actual_joint_pos: np.ndarray,
        target_joint_pos: np.ndarray,
    ) -> str:
        num_joints = actual_joint_pos.shape[1]
        columns = min(4, max(1, num_joints))
        rows = math.ceil(num_joints / columns)

        panel_width = 320
        panel_height = 190
        gutter_x = 18
        gutter_y = 20
        margin_x = 20
        header_height = 72
        footer_height = 20

        width = margin_x * 2 + columns * panel_width + (columns - 1) * gutter_x
        height = (
            header_height
            + rows * panel_height
            + max(0, rows - 1) * gutter_y
            + footer_height
        )

        t0 = float(time_s[0])
        t1 = float(time_s[-1])
        t_span = max(t1 - t0, 1e-6)
        sampled_idx = self._sample_indices(time_s.shape[0])
        sampled_time = time_s[sampled_idx]

        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            (
                f'<text x="{margin_x}" y="28" font-size="22" font-family="sans-serif" '
                f'font-weight="700" fill="#111827">{escape(self.title)}</text>'
            ),
            (
                f'<text x="{margin_x}" y="50" font-size="13" font-family="sans-serif" '
                f'fill="#4b5563">Blue: actual joint position, Orange: target joint position, '
                f'Samples: {time_s.shape[0]}, Duration: {t1 - t0:.2f} s</text>'
            ),
            (
                f'<line x1="{margin_x}" y1="58" x2="{width - margin_x}" y2="58" '
                'stroke="#e5e7eb" stroke-width="1"/>'
            ),
        ]

        for joint_idx, joint_name in enumerate(self.joint_names):
            row = joint_idx // columns
            col = joint_idx % columns
            panel_x = margin_x + col * (panel_width + gutter_x)
            panel_y = header_height + row * (panel_height + gutter_y)

            plot_left = panel_x + 46
            plot_top = panel_y + 30
            plot_width = panel_width - 58
            plot_height = panel_height - 58

            joint_actual = actual_joint_pos[:, joint_idx]
            joint_target = target_joint_pos[:, joint_idx]
            all_values = np.concatenate([joint_actual, joint_target], axis=0)
            y_min = float(np.min(all_values))
            y_max = float(np.max(all_values))
            if math.isclose(y_min, y_max, rel_tol=1e-6, abs_tol=1e-6):
                pad = max(0.05, abs(y_min) * 0.05)
            else:
                pad = max(0.05, (y_max - y_min) * 0.08)
            y_min -= pad
            y_max += pad
            y_span = max(y_max - y_min, 1e-6)

            def x_coord(t: float) -> float:
                return plot_left + ((t - t0) / t_span) * plot_width

            def y_coord(v: float) -> float:
                return plot_top + (1.0 - ((v - y_min) / y_span)) * plot_height

            actual_points = " ".join(
                f"{x_coord(float(sampled_time[i])):.2f},{y_coord(float(joint_actual[sampled_idx[i]])):.2f}"
                for i in range(sampled_idx.shape[0])
            )
            target_points = " ".join(
                f"{x_coord(float(sampled_time[i])):.2f},{y_coord(float(joint_target[sampled_idx[i]])):.2f}"
                for i in range(sampled_idx.shape[0])
            )

            mid_value = 0.5 * (y_min + y_max)
            mid_y = y_coord(mid_value)

            parts.extend([
                (
                    f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" '
                    f'height="{panel_height}" rx="8" fill="#f9fafb" stroke="#d1d5db"/>'
                ),
                (
                    f'<text x="{panel_x + 12}" y="{panel_y + 18}" font-size="13" '
                    f'font-family="sans-serif" font-weight="700" fill="#111827">'
                    f'{escape(joint_name)}</text>'
                ),
                (
                    f'<text x="{panel_x + 12}" y="{panel_y + 35}" font-size="11" '
                    f'font-family="sans-serif" fill="#6b7280">'
                    f'range [{y_min:.3f}, {y_max:.3f}] rad</text>'
                ),
                (
                    f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" '
                    f'height="{plot_height}" fill="#ffffff" stroke="#d1d5db"/>'
                ),
                (
                    f'<line x1="{plot_left}" y1="{mid_y:.2f}" x2="{plot_left + plot_width}" '
                    f'y2="{mid_y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
                ),
                (
                    f'<line x1="{plot_left}" y1="{plot_top + plot_height}" '
                    f'x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" '
                    'stroke="#9ca3af" stroke-width="1"/>'
                ),
                (
                    f'<polyline fill="none" stroke="#2563eb" stroke-width="1.5" '
                    f'points="{actual_points}"/>'
                ),
                (
                    f'<polyline fill="none" stroke="#f97316" stroke-width="1.5" '
                    f'points="{target_points}"/>'
                ),
                (
                    f'<text x="{plot_left - 6}" y="{plot_top + 4}" font-size="10" '
                    f'text-anchor="end" font-family="sans-serif" fill="#6b7280">{y_max:.2f}</text>'
                ),
                (
                    f'<text x="{plot_left - 6}" y="{mid_y + 4:.2f}" font-size="10" '
                    f'text-anchor="end" font-family="sans-serif" fill="#6b7280">{mid_value:.2f}</text>'
                ),
                (
                    f'<text x="{plot_left - 6}" y="{plot_top + plot_height + 4}" font-size="10" '
                    f'text-anchor="end" font-family="sans-serif" fill="#6b7280">{y_min:.2f}</text>'
                ),
                (
                    f'<text x="{plot_left}" y="{plot_top + plot_height + 16}" font-size="10" '
                    f'font-family="sans-serif" fill="#6b7280">{t0:.2f}s</text>'
                ),
                (
                    f'<text x="{plot_left + plot_width}" y="{plot_top + plot_height + 16}" '
                    f'font-size="10" text-anchor="end" font-family="sans-serif" fill="#6b7280">{t1:.2f}s</text>'
                ),
            ])

        parts.append("</svg>")
        return "\n".join(parts)
