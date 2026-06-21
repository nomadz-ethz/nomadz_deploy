"""RoboCup football-field + ball MuJoCo scene composer.

Dimensions follow the RoboCup Humanoid Soccer League rule book, 2026 v1.0
(release tag ``rules-2026-v1.0``, 2026-05-26) — the unified league linked from
https://humanoid.robocup.org/materials/rules/ and maintained at
https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules:

- Table 2 — field dimensions per field type (S-/M-/L-Field presets below).
- Table 3 — allowed goal dimension ranges per division (presets pick one
  in-range value each).
- Table 4 — balls per division: Small = FIFA Mini, Middle = FIFA size 3 or 4,
  Large = FIFA size 5.

The Middle Division (robot height < 1.25 m, so Booster K1) plays on the
M-Field: 14 x 9 m, goals 2.4-2.6 m wide x 1.5-1.9 m high, FIFA size 3/4 ball.

``compose_field_scene()`` parses a robot MJCF (e.g. ``K1_22dof_fixed.xml``),
removes its default flat ground (geom ``ground`` + ``texplane``/``matplane``
assets) and returns a self-contained scene XML string with:

- a green turf plane still named ``ground`` (MujocoController overrides
  friction/condim on that geom by name);
- white field markings as visual-only geoms (``contype=0 conaffinity=0``):
  touch/goal/halfway lines, center circle and marks, goal and penalty areas,
  penalty marks, corner arcs;
- two goals per the division spec (posts + crossbar + translucent net
  panels, all collidable);
- a free-joint ball sized per FIFA spec (contact tuning supplied by the
  task, since it must match what the policy was trained against).

Field frame: +X runs along the field length (goals at x = +-length/2),
+Y along the width, origin at the center mark. All rule measurements are
taken from the *outside* of the lines (lines belong to the area they bound).
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RoboCupFieldSpec:
    """One column of rule-book Table 2 (+ goal height from Table 3), meters."""

    name: str
    field_length: float          # A
    field_width: float           # B
    goal_depth: float            # C (range in rules)
    goal_width: float            # D, between the posts' inner faces (range)
    goal_area_length: float      # E
    goal_area_width: float       # F
    penalty_area_length: float   # G
    penalty_area_width: float    # H
    penalty_mark_distance: float  # I, goal-line outer edge -> mark center
    center_circle_diameter: float  # J, to the outside of the circle line
    border_strip_width: float    # K (rules minimum)
    corner_arc_radius: float     # L, 0 = no corner arcs (S-Field)
    line_width: float
    mark_size: float             # penalty/center mark edge length
    goal_height: float           # Table 3, ground -> underside of crossbar
    post_diameter: float = 0.10  # posts/crossbar width, rules allow 0.07-0.12


S_FIELD = RoboCupFieldSpec(
    name="S-Field (HSL 2026 Small Division)",
    field_length=9.0,
    field_width=6.0,
    goal_depth=0.6,
    goal_width=1.8,
    goal_area_length=1.0,
    goal_area_width=3.0,
    penalty_area_length=2.0,
    penalty_area_width=4.0,
    penalty_mark_distance=1.5,
    center_circle_diameter=1.5,
    border_strip_width=1.0,
    corner_arc_radius=0.0,
    line_width=0.05,
    mark_size=0.10,
    goal_height=1.2,
)

M_FIELD = RoboCupFieldSpec(
    name="M-Field (HSL 2026 Middle Division)",
    field_length=14.0,
    field_width=9.0,
    goal_depth=1.0,
    goal_width=2.6,
    goal_area_length=1.0,
    goal_area_width=4.0,
    penalty_area_length=3.0,
    penalty_area_width=6.0,
    penalty_mark_distance=2.0,
    center_circle_diameter=3.0,
    border_strip_width=1.0,
    corner_arc_radius=0.5,
    line_width=0.05,
    mark_size=0.10,
    goal_height=1.8,
)

L_FIELD = RoboCupFieldSpec(
    name="L-Field (HSL 2026 Large Division)",
    field_length=22.0,
    field_width=14.0,
    goal_depth=1.5,
    goal_width=3.0,
    goal_area_length=1.0,
    goal_area_width=5.0,
    penalty_area_length=3.5,
    penalty_area_width=7.0,
    penalty_mark_distance=2.5,
    center_circle_diameter=4.0,
    border_strip_width=1.0,
    corner_arc_radius=1.0,
    line_width=0.12,
    mark_size=0.15,
    goal_height=2.0,
)


@dataclass(frozen=True)
class BallSpec:
    """FIFA ball; radius from mid-range circumference, mass mid FIFA range."""

    name: str
    radius: float
    mass: float


FIFA_MINI = BallSpec("FIFA Mini (size 1)", radius=0.068, mass=0.205)
FIFA_SIZE_3 = BallSpec("FIFA size 3", radius=0.094, mass=0.31)    # 58-60 cm, 300-320 g
FIFA_SIZE_4 = BallSpec("FIFA size 4", radius=0.1025, mass=0.37)   # 63.5-66 cm, 350-390 g
FIFA_SIZE_5 = BallSpec("FIFA size 5", radius=0.11, mass=0.43)     # 68-70 cm, 410-450 g


# Marking boxes are a few tenths of a millimetre proud of the turf, on
# staggered z-tiers so that geoms which overlap (circle vs. halfway line,
# arcs vs. boundary lines, the two arms of a mark cross) never share a
# coplanar top face, which would z-fight in the viewer.
_LINE_HH = 0.001     # half-height of marking boxes
_Z_LINES = 0.0008    # straight lines (tiled edge-to-edge, no overlaps)
_Z_RING = 0.0016     # circle / corner-arc segments...
_Z_RING_STAGGER = 0.0004  # ...alternating, as neighbouring segments overlap
_Z_MARK = 0.0026     # mark crosses, first arm
_Z_MARK2 = 0.0031    # mark crosses, second arm

_LINE_RGBA = "1 1 1 1"
# The net is drawn as a grid of thin visual-only strands; the ball is caught
# by invisible collidable panels spanning the same faces.
_NET_STRAND_RADIUS = 0.006
_NET_MESH_SIZE = 0.13
_NET_STRAND_RGBA = "0.92 0.92 0.92 1"
_NET_PANEL_RGBA = "1 1 1 0"


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def _yaw_quat(yaw: float) -> str:
    # quat instead of euler so the fragment is unit-independent of the
    # robot file's <compiler angle=...> setting.
    return f"{_fmt(math.cos(yaw / 2.0))} 0 0 {_fmt(math.sin(yaw / 2.0))}"


def _box(
    x: float,
    y: float,
    z: float,
    hx: float,
    hy: float,
    hz: float,
    *,
    rgba: str = _LINE_RGBA,
    yaw: float | None = None,
    name: str | None = None,
    collidable: bool = False,
) -> str:
    attrs = []
    if name:
        attrs.append(f'name="{name}"')
    attrs.append('type="box"')
    attrs.append(f'pos="{_fmt(x)} {_fmt(y)} {_fmt(z)}"')
    attrs.append(f'size="{_fmt(hx)} {_fmt(hy)} {_fmt(hz)}"')
    if yaw is not None:
        attrs.append(f'quat="{_yaw_quat(yaw)}"')
    if not collidable:
        attrs.append('contype="0" conaffinity="0"')
    attrs.append(f'rgba="{rgba}"')
    return f"<geom {' '.join(attrs)}/>"


def _ring(
    cx: float,
    cy: float,
    radius: float,
    a_start: float,
    a_end: float,
    nseg: int,
    line_width: float,
) -> list[str]:
    """A circular-arc line (centerline ``radius``) as thin tangent boxes."""
    da = (a_end - a_start) / nseg
    # Slight overlap closes the chord-vs-arc gaps; the z stagger below keeps
    # the overlapping tops from z-fighting.
    half_len = radius * math.tan(abs(da) / 2.0) + line_width / 2.0
    geoms = []
    for i in range(nseg):
        a = a_start + (i + 0.5) * da
        z = _Z_RING + (_Z_RING_STAGGER if i % 2 else 0.0)
        geoms.append(
            _box(
                cx + radius * math.cos(a),
                cy + radius * math.sin(a),
                z,
                half_len,
                line_width / 2.0,
                _LINE_HH,
                yaw=a + math.pi / 2.0,
            )
        )
    return geoms


def _capsule(p0, p1, radius: float, rgba: str) -> str:
    return (
        f'<geom type="capsule" size="{_fmt(radius)}" '
        f'fromto="{_fmt(p0[0])} {_fmt(p0[1])} {_fmt(p0[2])} '
        f'{_fmt(p1[0])} {_fmt(p1[1])} {_fmt(p1[2])}" '
        f'contype="0" conaffinity="0" rgba="{rgba}"/>'
    )


def _net_offsets(length: float, interior_only: bool = False) -> list[float]:
    n = max(1, round(length / _NET_MESH_SIZE))
    rng = range(1, n) if interior_only else range(0, n + 1)
    return [k / n for k in rng]


def _net_face(origin, u, v, u_offsets, v_offsets) -> list[str]:
    """Net strands over the rectangle ``origin + a*u + b*v`` (a, b in [0,1]).

    Strands parallel to ``u`` are drawn at fractions ``b`` in ``v_offsets``,
    strands parallel to ``v`` at fractions ``a`` in ``u_offsets``. Offsets are
    fractional so callers can drop boundary strands that an adjacent face or
    the goal frame already provides (coincident strands would z-fight).
    """

    def at(a: float, b: float):
        return (
            origin[0] + a * u[0] + b * v[0],
            origin[1] + a * u[1] + b * v[1],
            origin[2] + a * u[2] + b * v[2],
        )

    geoms = []
    for b in v_offsets:
        geoms.append(_capsule(at(0.0, b), at(1.0, b), _NET_STRAND_RADIUS, _NET_STRAND_RGBA))
    for a in u_offsets:
        geoms.append(_capsule(at(a, 0.0), at(a, 1.0), _NET_STRAND_RADIUS, _NET_STRAND_RGBA))
    return geoms


def _mark_cross(x: float, y: float, spec: RoboCupFieldSpec) -> list[str]:
    """Penalty/center mark: a cross of ``mark_size`` length, line width arms."""
    arm = spec.mark_size / 2.0
    lw_h = spec.line_width / 2.0
    return [
        _box(x, y, _Z_MARK, arm, lw_h, _LINE_HH),
        _box(x, y, _Z_MARK2, lw_h, arm, _LINE_HH),
    ]


def _area_lines(sign: float, length: float, width: float, spec: RoboCupFieldSpec) -> list[str]:
    """Goal-area / penalty-area box at the ``sign``-x end of the field.

    ``length`` x ``width`` measured to the outside of the lines: the outer
    rectangle spans x in [half_l - length, half_l], y in [-width/2, width/2].
    The three lines tile edge-to-edge with the goal line (no overlaps).
    """
    half_l = spec.field_length / 2.0
    lw = spec.line_width
    lw_h = lw / 2.0
    front_outer = half_l - length
    geoms = [
        # Line parallel to the goal line, full area width.
        _box(sign * (front_outer + lw_h), 0.0, _Z_LINES, lw_h, width / 2.0, _LINE_HH)
    ]
    # Side lines, from the front line's inner edge to the goal line's inner edge.
    lo, hi = front_outer + lw, half_l - lw
    cx, hx = (lo + hi) / 2.0, (hi - lo) / 2.0
    for sy in (1.0, -1.0):
        geoms.append(
            _box(sign * cx, sy * (width / 2.0 - lw_h), _Z_LINES, hx, lw_h, _LINE_HH)
        )
    return geoms


def _goal(sign: float, spec: RoboCupFieldSpec) -> list[str]:
    """Goal at the ``sign``-x end: posts + crossbar + translucent net panels."""
    half_l = spec.field_length / 2.0
    pr = spec.post_diameter / 2.0
    x_line = sign * (half_l - spec.line_width / 2.0)  # posts centered on the goal line
    y_post = spec.goal_width / 2.0 + pr               # goal_width = between inner faces
    z_bar = spec.goal_height + pr                     # crossbar underside at goal_height
    z_top = z_bar + pr
    tag = "xp" if sign > 0 else "xn"
    geoms = []
    for sy, side in ((1.0, "l"), (-1.0, "r")):
        geoms.append(
            f'<geom name="goal_{tag}_post_{side}" type="cylinder" size="{_fmt(pr)}" '
            f'fromto="{_fmt(x_line)} {_fmt(sy * y_post)} 0 '
            f'{_fmt(x_line)} {_fmt(sy * y_post)} {_fmt(z_top)}" rgba="{_LINE_RGBA}"/>'
        )
    geoms.append(
        f'<geom name="goal_{tag}_crossbar" type="cylinder" size="{_fmt(pr)}" '
        f'fromto="{_fmt(x_line)} {_fmt(-(y_post + pr))} {_fmt(z_bar)} '
        f'{_fmt(x_line)} {_fmt(y_post + pr)} {_fmt(z_bar)}" rgba="{_LINE_RGBA}"/>'
    )
    # Ball containment: invisible collidable panels (back, two sides, top);
    # what the eye sees is the strand grid drawn below on the same faces.
    t = 0.01
    hy = y_post + pr
    hz = z_top / 2.0
    x_mid = x_line + sign * spec.goal_depth / 2.0
    x_back = x_line + sign * spec.goal_depth
    geoms.append(
        _box(x_back, 0.0, hz, t, hy, hz,
             rgba=_NET_PANEL_RGBA, name=f"goal_{tag}_net_back", collidable=True)
    )
    for sy, side in ((1.0, "l"), (-1.0, "r")):
        geoms.append(
            _box(x_mid, sy * hy, hz, spec.goal_depth / 2.0, t, hz,
                 rgba=_NET_PANEL_RGBA, name=f"goal_{tag}_net_{side}", collidable=True)
        )
    geoms.append(
        _box(x_mid, 0.0, z_top - t, spec.goal_depth / 2.0, hy, t,
             rgba=_NET_PANEL_RGBA, name=f"goal_{tag}_net_top", collidable=True)
    )

    # Visible net: strand grids on the back, side and top faces. Boundary
    # strands are dropped where an adjacent face or the goal frame (posts,
    # crossbar, back-face edges) already runs along the same edge.
    span_y = 2.0 * hy
    depth_vec = (x_back - x_line, 0.0, 0.0)
    geoms += _net_face(
        (x_back, -hy, 0.0), (0.0, span_y, 0.0), (0.0, 0.0, z_top),
        _net_offsets(span_y), _net_offsets(z_top),
    )
    for sy in (1.0, -1.0):
        geoms += _net_face(
            (x_line, sy * hy, 0.0), depth_vec, (0.0, 0.0, z_top),
            _net_offsets(spec.goal_depth, interior_only=True), _net_offsets(z_top),
        )
    geoms += _net_face(
        (x_line, -hy, z_top), depth_vec, (0.0, span_y, 0.0),
        _net_offsets(spec.goal_depth, interior_only=True),
        _net_offsets(span_y, interior_only=True),
    )
    return geoms


def _field_assets_xml() -> str:
    return (
        '<texture name="turf_tex" type="2d" builtin="checker" '
        'rgb1="0.255 0.482 0.204" rgb2="0.224 0.435 0.176" width="512" height="512"/>'
        '<material name="turf_mat" texture="turf_tex" texrepeat="0.5 0.5" '
        'texuniform="true" reflectance="0.05"/>'
        '<texture name="ball_tex" type="cube" builtin="checker" '
        'rgb1="0.95 0.95 0.95" rgb2="0.12 0.12 0.12" width="128" height="128"/>'
        '<material name="ball_mat" texture="ball_tex" reflectance="0.02"/>'
    )


def _field_worldbody_xml(spec: RoboCupFieldSpec) -> str:
    half_l = spec.field_length / 2.0
    half_w = spec.field_width / 2.0
    lw = spec.line_width
    lw_h = lw / 2.0

    # Turf extends one border strip beyond the lines; deep goals (L-Field)
    # need a little more so the net does not hang off the rendered carpet.
    border = max(spec.border_strip_width, spec.goal_depth + 0.3)
    geoms = [
        # The walking surface. Named "ground" so MujocoController's
        # name-based condim/friction override keeps working; collision is
        # infinite (plane), only the rendered rectangle is finite.
        f'<geom name="ground" type="plane" pos="0 0 0" '
        f'size="{_fmt(half_l + border)} {_fmt(half_w + border)} 0.1" '
        f'material="turf_mat" condim="1" friction="0.4 0.005 0.0001"/>'
    ]

    # Boundary + halfway lines, tiled edge-to-edge: goal lines span the full
    # width, touchlines run between their inner edges, the halfway line runs
    # between the touchlines' inner edges.
    for sx in (1.0, -1.0):
        geoms.append(
            _box(sx * (half_l - lw_h), 0.0, _Z_LINES, lw_h, half_w, _LINE_HH)
        )
    for sy in (1.0, -1.0):
        geoms.append(
            _box(0.0, sy * (half_w - lw_h), _Z_LINES, half_l - lw, lw_h, _LINE_HH)
        )
    geoms.append(_box(0.0, 0.0, _Z_LINES, lw_h, half_w - lw, _LINE_HH))

    # Center circle (diameter to the outside of the line) + center mark; the
    # halfway line already provides the cross's y-stroke at the center.
    geoms += _ring(0.0, 0.0, spec.center_circle_diameter / 2.0 - lw_h,
                   0.0, 2.0 * math.pi, 48, lw)
    geoms.append(_box(0.0, 0.0, _Z_MARK, spec.mark_size / 2.0, lw_h, _LINE_HH))

    for sx in (1.0, -1.0):
        geoms += _area_lines(sx, spec.goal_area_length, spec.goal_area_width, spec)
        geoms += _area_lines(sx, spec.penalty_area_length, spec.penalty_area_width, spec)
        geoms += _mark_cross(sx * (half_l - spec.penalty_mark_distance), 0.0, spec)
        geoms += _goal(sx, spec)

    if spec.corner_arc_radius > 0.0:
        r = spec.corner_arc_radius - lw_h
        quarter = math.pi / 2.0
        for sx, sy in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            # Quarter circle centered on the corner, opening into the field.
            a_start = math.atan2(-sy, -sx) - quarter / 2.0
            geoms += _ring(sx * half_l, sy * half_w, r,
                           a_start, a_start + quarter, 8, lw)

    return "".join(geoms)


def _ball_xml(
    ball: BallSpec,
    body_name: str,
    contact_attrs: Mapping[str, str],
    init_xy: tuple[float, float],
) -> str:
    extra = " ".join(f'{k}="{v}"' for k, v in contact_attrs.items())
    return (
        f'<body name="{body_name}" '
        f'pos="{_fmt(init_xy[0])} {_fmt(init_xy[1])} {_fmt(ball.radius)}">'
        f'<freejoint name="{body_name}_joint"/>'
        f'<geom name="{body_name}_geom" type="sphere" size="{_fmt(ball.radius)}" '
        f'mass="{_fmt(ball.mass)}" material="ball_mat" {extra}/>'
        f"</body>"
    )


def compose_field_scene(
    robot_mjcf_path: str,
    *,
    model_name: str,
    field: RoboCupFieldSpec = M_FIELD,
    ball: BallSpec = FIFA_SIZE_3,
    ball_contact_attrs: Mapping[str, str] | None = None,
    ball_body_name: str = "ball",
    ball_init_xy: tuple[float, float] = (1.0, 0.0),
) -> str:
    """Compose robot + RoboCup field + ball into one scene XML string.

    The robot file is inlined (not ``<include>``-ed) so its default flat
    ground and the associated checker-plane assets can be removed. The ball
    body/joint/geom are named ``{ball_body_name}``/``..._joint``/``..._geom``
    to match what DribblingPolicy looks up. ``ball_contact_attrs`` (friction/
    condim/solimp/solref strings) come from the task so contact behaviour
    stays whatever the policy was tuned against. The initial ball pose is a
    placeholder — every reset re-spawns it via the policy.
    """
    tree = ET.parse(robot_mjcf_path)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError(f"{robot_mjcf_path} is not an MJCF file")
    root.set("model", model_name)

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for el in list(asset):
        if el.get("name") in ("texplane", "matplane"):
            asset.remove(el)

    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"{robot_mjcf_path} has no <worldbody>")
    for el in list(world):
        if el.tag == "geom" and el.get("name") == "ground":
            world.remove(el)

    fragments = (
        (asset, _field_assets_xml()),
        (world, _field_worldbody_xml(field)),
        (world, _ball_xml(ball, ball_body_name,
                          dict(ball_contact_attrs or {}), ball_init_xy)),
    )
    for parent, xml in fragments:
        for child in ET.fromstring(f"<wrap>{xml}</wrap>"):
            parent.append(child)

    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
    header = (
        f"<!-- Generated scene: do not edit by hand. "
        f"{os.path.basename(robot_mjcf_path)} + {field.name} + {ball.name}; "
        f"regenerated on task import when the composer inputs change. -->\n"
    )
    return header + ET.tostring(root, encoding="unicode") + "\n"
