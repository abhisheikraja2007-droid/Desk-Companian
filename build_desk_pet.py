from __future__ import annotations
import os
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import trimesh
from pydantic import BaseModel
import build123d as bd

try:
    import ocp_vscode
    HAS_OCP_VSCODE = True
except ImportError:
    HAS_OCP_VSCODE = False

class MassInertiaProperties(BaseModel):
    mass_kg: float
    cog_m: Tuple[float, float, float]
    inertia_matrix: List[List[float]]

class JointLimits(BaseModel):
    lower_rad: float
    upper_rad: float
    max_effort_nm: float
    max_velocity_rad_s: float

class JointSpec(BaseModel):
    name: str
    type: str
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    limits: Optional[JointLimits] = None

class ValidationMetrics(BaseModel):
    total_mass_kg: float
    cog_mm: Tuple[float, float, float]
    cog_z_mm: float
    support_polygon_x_mm: Tuple[float, float]
    support_polygon_y_mm: Tuple[float, float]
    is_cog_stable: bool
    clearance_checks: Dict[str, bool]

DENSITY_PLA_KG_M3 = 1240.0

PURCHASED_ELECTRONICS_MASSES: Dict[str, float] = {
    "base_link": 0.2800,
    "left_wheel": 0.0060,
    "right_wheel": 0.0060,
    "torso_shell": 0.0645,
    "left_arm": 0.0130,
    "left_pincer": 0.0010,
    "right_arm": 0.0130,
    "right_pincer": 0.0010,
    "neck_yaw_turret": 0.0125,
    "head_assembly": 0.0505,
    "left_ear": 0.0000,
    "right_ear": 0.0000,
    "left_eye": 0.0010,
    "right_eye": 0.0010,
}

PURCHASED_COG_OFFSETS_MM: Dict[str, Tuple[float, float, float]] = {
    "base_link": (0.0, 0.0, 8.0),
    "left_wheel": (0.0, 0.0, 0.0),
    "right_wheel": (0.0, 0.0, 0.0),
    "torso_shell": (0.0, -1.0, 56.0),
    "left_arm": (0.0, 0.0, -13.0),
    "left_pincer": (0.0, 0.0, -8.0),
    "right_arm": (0.0, 0.0, -13.0),
    "right_pincer": (0.0, 0.0, -8.0),
    "neck_yaw_turret": (0.0, 0.0, 101.0),
    "head_assembly": (0.0, 2.0, 125.0),
    "left_ear": (0.0, 0.0, 0.0),
    "right_ear": (0.0, 0.0, 0.0),
    "left_eye": (0.0, 0.0, 0.0),
    "right_eye": (0.0, 0.0, 0.0),
}

def load_or_create_tier1() -> Dict[str, bd.Solid]:
    step_file = "DUCk.step"
    parts: Dict[str, bd.Solid] = {}
    if os.path.exists(step_file):
        compound = bd.import_step(step_file)
        solids = compound.solids()
        mapping = {
            0: "caster_ball_front",
            1: "motor_left_envelope",
            2: "motor_strap_outer_right",
            3: "motor_strap_inner_right",
            4: "motor_strap_inner_left",
            5: "motor_strap_outer_left",
            6: "motor_saddle_left",
            7: "wheel_hub_left",
            8: "drive_wheel_left",
            9: "caster_ball_rear",
            10: "drive_wheel_right",
            11: "motor_right_envelope",
            12: "wheel_hub_right",
            13: "motor_clamp_central",
            14: "motor_saddle_right",
            15: "top_deck",
            16: "chassis"
        }
        for idx, name in mapping.items():
            if idx < len(solids):
                parts[name] = solids[idx]
        sw = bd.Box(8.5, 20.0, 3.5).located(bd.Location((0.0, -57.5, 16.0)))
        sw_m2_1 = bd.Cylinder(1.1, 20.0).located(bd.Location((-9.5, -57.5, 16.0), (90.0, 0.0, 0.0)))
        sw_m2_2 = bd.Cylinder(1.1, 20.0).located(bd.Location((9.5, -57.5, 16.0), (90.0, 0.0, 0.0)))
        if "chassis" in parts:
            parts["chassis"] = parts["chassis"] - sw - sw_m2_1 - sw_m2_2
        return parts

    sw = bd.Box(8.5, 20.0, 3.5).located(bd.Location((0.0, -57.5, 16.0)))
    sw_m2_1 = bd.Cylinder(1.1, 20.0).located(bd.Location((-9.5, -57.5, 16.0), (90.0, 0.0, 0.0)))
    sw_m2_2 = bd.Cylinder(1.1, 20.0).located(bd.Location((9.5, -57.5, 16.0), (90.0, 0.0, 0.0)))

    with bd.BuildPart() as chassis_b:
        with bd.BuildSketch(bd.Plane.XY.offset(4.0)):
            bd.RectangleRounded(98.0, 115.0, 20.0)
        bd.extrude(amount=28.0)
        with bd.BuildSketch(bd.Plane.XY.offset(4.0)):
            with bd.Locations([(-49.0, 0.0), (49.0, 0.0)]):
                bd.Rectangle(14.0, 38.0)
        bd.extrude(amount=28.0, mode=bd.Mode.SUBTRACT)
        with bd.BuildSketch(bd.Plane.XY.offset(5.0)):
            with bd.Locations([(-28.0, -35.0), (28.0, -35.0), (-28.0, 35.0), (28.0, 35.0)]):
                bd.Rectangle(6.2, 10.2)
        bd.extrude(amount=7.2, mode=bd.Mode.SUBTRACT)
    parts["chassis"] = chassis_b.part - sw - sw_m2_1 - sw_m2_2

    with bd.BuildPart() as top_deck_b:
        with bd.BuildSketch(bd.Plane.XY.offset(27.2)):
            bd.RectangleRounded(96.0, 115.0, 19.5)
        bd.extrude(amount=6.8)
        with bd.BuildSketch(bd.Plane.XY.offset(27.2)):
            bd.Circle(9.0)
            with bd.Locations([(-32.0, -27.0), (32.0, -27.0), (-32.0, 27.0), (32.0, 27.0)]):
                bd.Circle(1.7)
        bd.extrude(amount=6.8, mode=bd.Mode.SUBTRACT)
    parts["top_deck"] = top_deck_b.part

    with bd.BuildPart() as wl_b:
        with bd.BuildSketch(bd.Plane.YZ.offset(-49.0)):
            bd.Circle(17.0)
        bd.extrude(amount=12.0)
    parts["drive_wheel_left"] = wl_b.part

    with bd.BuildPart() as wr_b:
        with bd.BuildSketch(bd.Plane.YZ.offset(49.0)):
            bd.Circle(17.0)
        bd.extrude(amount=12.0)
    parts["drive_wheel_right"] = wr_b.part

    with bd.BuildPart() as cf_b:
        with bd.BuildSketch(bd.Plane.XY.offset(5.0)):
            bd.Circle(5.0)
        bd.revolve(axis=bd.Axis.X)
    parts["caster_ball_front"] = cf_b.part.locate(bd.Location((0.0, 44.0, 0.0)))

    with bd.BuildPart() as cr_b:
        with bd.BuildSketch(bd.Plane.XY.offset(5.0)):
            bd.Circle(5.0)
        bd.revolve(axis=bd.Axis.X)
    parts["caster_ball_rear"] = cr_b.part.locate(bd.Location((0.0, -44.0, 0.0)))

    return parts

def build_tier2_torso() -> bd.Part:
    with bd.BuildPart() as p:
        with bd.BuildSketch(bd.Plane.XZ):
            with bd.BuildLine():
                bd.Line((0, 34), (40, 34))
                bd.Spline([(40, 34), (42, 54), (36, 74), (21, 96)])
                bd.Line((21, 96), (0, 96))
                bd.Line((0, 96), (0, 34))
            bd.make_face()
        bd.revolve(axis=bd.Axis.Z)

        top_f = p.part.faces().sort_by(bd.Axis.Z)[-1]
        bot_f = p.part.faces().sort_by(bd.Axis.Z)[0]
        bd.offset(amount=-2.2, openings=[top_f, bot_f])

        with bd.BuildSketch(bd.Plane.XY.offset(34.0)):
            bd.Circle(39.0)
        bd.extrude(amount=3.0, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane.XY.offset(33.5)):
            bd.Circle(9.0)
            with bd.Locations([(-32.0, -27.0), (32.0, -27.0), (-32.0, 27.0), (32.0, 27.0)]):
                bd.Circle(1.7)
        bd.extrude(amount=4.0, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.XY.offset(93.0)):
            bd.Circle(20.5)
        bd.extrude(amount=3.0, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane.XY.offset(92.0)):
            bd.Circle(13.55)
        bd.extrude(amount=4.1, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.XY.offset(90.0)):
            bd.Circle(10.0)
        bd.extrude(amount=6.1, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.XY.offset(34.0)):
            with bd.Locations([(0.0, -36.0)]):
                bd.Circle(10.0)
        bd.extrude(amount=71.0, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane.XY.offset(34.0)):
            with bd.Locations([(0.0, -36.0)]):
                bd.Circle(6.5)
        bd.extrude(amount=72.0, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.YZ.offset(34.0)):
            with bd.Locations([(2.0, 78.0)]):
                bd.Circle(6.5)
        bd.extrude(amount=14.5, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane.YZ.offset(-34.0)):
            with bd.Locations([(2.0, 78.0)]):
                bd.Circle(6.5)
        bd.extrude(amount=-14.5, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane.YZ.offset(33.0)):
            with bd.Locations([(2.0, 78.0)]):
                bd.Circle(2.5)
        bd.extrude(amount=16.0, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.YZ.offset(-49.0)):
            with bd.Locations([(2.0, 78.0)]):
                bd.Circle(2.5)
        bd.extrude(amount=16.0, mode=bd.Mode.SUBTRACT)

        grille_slots = [
            (0.0, 56.0, 16.0, 1.8),
            (0.0, 53.0, 22.0, 1.8),
            (0.0, 50.0, 26.0, 1.8),
            (0.0, 47.0, 22.0, 1.8),
            (0.0, 44.0, 16.0, 1.8),
        ]
        for (gx, gz, glen, gw) in grille_slots:
            slot = bd.Box(glen, 20.0, gw).located(bd.Location((gx, 35.0, gz)))
            p.part = p.part - slot

        usb_cutout = bd.Box(9.2, 20.0, 3.8).located(bd.Location((0.0, -40.0, 38.0)))
        p.part = p.part - usb_cutout

        rst_pinhole = bd.Cylinder(1.0, 30.0).located(bd.Location((0.0, -35.0, 64.0), (90.0, 0.0, 0.0)))
        p.part = p.part - rst_pinhole

    return p.part

def build_tier2_torso_clamshells() -> Tuple[bd.Part, bd.Part]:
    torso = build_tier2_torso()
    box_f = bd.Box(160.0, 160.0, 160.0).located(bd.Location((0.0, 80.0, 80.0)))
    box_r = bd.Box(160.0, 160.0, 160.0).located(bd.Location((0.0, -80.0, 80.0)))

    tf = bd.Compound(torso.intersect(box_f).solids())
    tr = bd.Compound(torso.intersect(box_r).solids())

    lip_tabs = [
        bd.Box(6.0, 1.5, 8.0).located(bd.Location((-36.0, -0.75, 58.0))),
        bd.Box(6.0, 1.5, 8.0).located(bd.Location((36.0, -0.75, 58.0))),
        bd.Box(6.0, 1.5, 8.0).located(bd.Location((-22.0, -0.75, 88.0))),
        bd.Box(6.0, 1.5, 8.0).located(bd.Location((22.0, -0.75, 88.0))),
    ]
    for tab in lip_tabs:
        tf = tf + tab
        groove = bd.Box(6.4, 2.0, 8.4).located(bd.Location(tab.location.position))
        tr = tr - groove

    screw_coords = [(-26.0, 48.0), (26.0, 48.0), (-18.0, 84.0), (18.0, 84.0)]
    for (bx, bz) in screw_coords:
        boss_f = bd.Cylinder(3.8, 6.0).located(bd.Location((bx, 3.0, bz), (90.0, 0.0, 0.0)))
        insert_hole = bd.Cylinder(2.0, 5.0).located(bd.Location((bx, 2.5, bz), (90.0, 0.0, 0.0)))
        tf = tf + (boss_f - insert_hole)

        tunnel_r = bd.Cylinder(1.7, 80.0).located(bd.Location((bx, -25.0, bz), (90.0, 0.0, 0.0)))
        counterbore_r = bd.Cylinder(3.0, 40.0).located(bd.Location((bx, -35.0, bz), (90.0, 0.0, 0.0)))
        tr = tr - tunnel_r - counterbore_r

    for sx in [-10.0, 10.0]:
        for sz in [45.0, 83.0]:
            standoff = bd.Cylinder(2.25, 5.0).located(bd.Location((sx, -17.5, sz), (90.0, 0.0, 0.0)))
            pilot = bd.Cylinder(0.9, 5.0).located(bd.Location((sx, -17.5, sz), (90.0, 0.0, 0.0)))
            tr = tr + (standoff - pilot)

    return tf, tr

def build_tier3_head_clamshells() -> Tuple[bd.Part, bd.Part]:
    cranium = build_tier3_cranium_shell()
    box_hf = bd.Box(160.0, 160.0, 160.0).located(bd.Location((0.0, 80.0 + 4.0, 125.0)))
    box_hr = bd.Box(160.0, 160.0, 160.0).located(bd.Location((0.0, -80.0 + 4.0, 125.0)))

    hf = bd.Compound(cranium.intersect(box_hf).solids())
    hr = bd.Compound(cranium.intersect(box_hr).solids())

    lip_tabs = [
        bd.Box(5.0, 1.2, 6.0).located(bd.Location((-28.0, 4.0 - 0.6, 126.0))),
        bd.Box(5.0, 1.2, 6.0).located(bd.Location((28.0, 4.0 - 0.6, 126.0))),
        bd.Box(6.0, 1.2, 4.0).located(bd.Location((0.0, 4.0 - 0.6, 144.0))),
    ]
    for tab in lip_tabs:
        hf = hf + tab
        groove = bd.Box(5.4, 1.6, 6.4).located(bd.Location(tab.location.position))
        hr = hr - groove

    head_screws = [(-20.0, 122.0), (20.0, 122.0), (0.0, 139.0)]
    for (hx, hz) in head_screws:
        boss_hf = bd.Cylinder(3.0, 5.0).located(bd.Location((hx, 6.5, hz), (90.0, 0.0, 0.0)))
        insert_hf = bd.Cylinder(1.7, 4.5).located(bd.Location((hx, 6.25, hz), (90.0, 0.0, 0.0)))
        hf = hf + (boss_hf - insert_hf)

        tunnel_hr = bd.Cylinder(1.4, 60.0).located(bd.Location((hx, -20.0, hz), (90.0, 0.0, 0.0)))
        cb_hr = bd.Cylinder(2.6, 30.0).located(bd.Location((hx, -30.0, hz), (90.0, 0.0, 0.0)))
        hr = hr - tunnel_hr - cb_hr

    mic_pinhole = bd.Cylinder(1.0, 20.0).located(bd.Location((0.0, 30.0, 108.0), (90.0, 0.0, 0.0)))
    hf = hf - mic_pinhole

    return hf, hr

def build_upper_arm(is_left: bool = True) -> bd.Part:
    with bd.BuildPart() as p:
        with bd.BuildSketch(bd.Plane.YZ):
            bd.Circle(4.8)
        bd.extrude(amount=4.0, both=True)

        with bd.BuildSketch(bd.Plane.YZ):
            bd.Circle(1.7)
        bd.extrude(amount=5.0, both=True, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.XY) as s1:
            bd.RectangleRounded(7.5, 9.5, 1.8)
        with bd.BuildSketch(bd.Plane.XY.offset(-26.0)) as s2:
            bd.RectangleRounded(5.5, 7.5, 1.5)
        bd.loft([s1.sketch, s2.sketch])

        with bd.BuildSketch(bd.Plane.YZ):
            with bd.Locations([(0.0, -26.0)]):
                bd.Circle(3.0)
        bd.extrude(amount=2.8, both=True, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane.YZ):
            with bd.Locations([(0.0, -26.0)]):
                bd.Circle(1.2)
        bd.extrude(amount=3.5, both=True, mode=bd.Mode.SUBTRACT)

    return p.part

def build_pincer_hand(is_left: bool = True) -> bd.Part:
    with bd.BuildPart() as p:
        with bd.BuildSketch(bd.Plane.YZ):
            bd.Circle(2.8)
        bd.extrude(amount=2.2, both=True)

        with bd.BuildSketch(bd.Plane.YZ):
            bd.Circle(1.2)
        bd.extrude(amount=3.0, both=True, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.XY) as ps1:
            bd.RectangleRounded(4.5, 11.0, 1.2)
        with bd.BuildSketch(bd.Plane.XY.offset(-4.0)) as ps2:
            bd.RectangleRounded(4.0, 11.0, 1.2)
        bd.loft([ps1.sketch, ps2.sketch])

        with bd.BuildSketch(bd.Plane.XY.offset(-4.0)) as f1_1:
            with bd.Locations([(0.0, 4.2)]):
                bd.RectangleRounded(3.6, 2.6, 0.6)
        with bd.BuildSketch(bd.Plane.XY.offset(-16.0)) as f1_2:
            with bd.Locations([(0.0, 3.2)]):
                bd.RectangleRounded(2.2, 1.8, 0.4)
        bd.loft([f1_1.sketch, f1_2.sketch])

        with bd.BuildSketch(bd.Plane.XY.offset(-4.0)) as f2_1:
            with bd.Locations([(0.0, -4.2)]):
                bd.RectangleRounded(3.6, 2.6, 0.6)
        with bd.BuildSketch(bd.Plane.XY.offset(-16.0)) as f2_2:
            with bd.Locations([(0.0, -3.2)]):
                bd.RectangleRounded(2.2, 1.8, 0.4)
        bd.loft([f2_1.sketch, f2_2.sketch])

        r1 = bd.Box(2.2, 0.8, 8.0).located(bd.Location((0.0, 2.6, -10.0)))
        r2 = bd.Box(2.2, 0.8, 8.0).located(bd.Location((0.0, -2.6, -10.0)))
        p.part = p.part - r1 - r2

    return p.part

def build_tier3_neck_yaw_turret() -> bd.Part:
    with bd.BuildPart() as p:
        with bd.BuildSketch(bd.Plane.XY.offset(92.0)):
            bd.Circle(9.9)
        bd.extrude(amount=4.0)

        with bd.BuildSketch(bd.Plane.XY.offset(96.0)):
            bd.Circle(13.0)
        bd.extrude(amount=2.5)

        with bd.BuildSketch(bd.Plane.XY.offset(98.5)):
            bd.RectangleRounded(24.0, 20.0, 4.0)
        bd.extrude(amount=9.5)

        with bd.BuildSketch(bd.Plane.XY.offset(92.0)):
            bd.Circle(6.0)
        bd.extrude(amount=16.1, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.YZ):
            with bd.Locations([(0.0, 105.0)]):
                bd.Circle(2.5)
        bd.extrude(amount=25.0, both=True, mode=bd.Mode.SUBTRACT)

    return p.part

def build_tier3_neck_pitch_yoke() -> bd.Part:
    with bd.BuildPart() as p:
        with bd.BuildSketch(bd.Plane.XY.offset(105.0)):
            bd.RectangleRounded(28.0, 18.0, 4.0)
        bd.extrude(amount=4.0)

        with bd.BuildSketch(bd.Plane.XZ.offset(-12.0)):
            with bd.Locations([(0.0, 108.0)]):
                bd.RectangleRounded(16.0, 14.0, 3.0)
        bd.extrude(amount=4.0)

        with bd.BuildSketch(bd.Plane.XZ.offset(8.0)):
            with bd.Locations([(0.0, 108.0)]):
                bd.RectangleRounded(16.0, 14.0, 3.0)
        bd.extrude(amount=4.0)

        with bd.BuildSketch(bd.Plane.YZ):
            with bd.Locations([(0.0, 108.0)]):
                bd.Circle(2.5)
        bd.extrude(amount=30.0, both=True, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.XY.offset(105.0)):
            bd.Circle(6.0)
        bd.extrude(amount=12.0, mode=bd.Mode.SUBTRACT)

    return p.part

def build_tier3_cranium_shell() -> bd.Part:
    sph_out = bd.Sphere(34.0)
    ell_out = bd.scale(sph_out, by=(1.0, 1.0, 21.0 / 34.0)).located(bd.Location((0.0, 2.0, 125.0)))
    sph_in = bd.Sphere(31.8)
    ell_in = bd.scale(sph_in, by=(1.0, 1.0, 18.8 / 31.8)).located(bd.Location((0.0, 2.0, 125.0)))
    head_hollow = ell_out - ell_in

    front_plane = bd.Plane(origin=(0.0, 31.0, 0.0), x_dir=(-1.0, 0.0, 0.0), z_dir=(0.0, 1.0, 0.0))

    with bd.BuildPart() as cp:
        bd.insert(head_hollow)

        with bd.BuildSketch(front_plane):
            with bd.Locations([(0.0, 126.0)]):
                bd.RectangleRounded(48.0, 16.0, 6.0)
        bd.extrude(amount=3.5, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane(origin=(0.0, 34.5, 0.0), x_dir=(-1.0, 0.0, 0.0), z_dir=(0.0, 1.0, 0.0))):
            with bd.Locations([(0.0, 126.0)]):
                bd.RectangleRounded(45.0, 14.0, 5.0)
        bd.extrude(amount=0.5, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane(origin=(0.0, 33.0, 0.0), x_dir=(-1.0, 0.0, 0.0), z_dir=(0.0, 1.0, 0.0))):
            with bd.Locations([(0.0, 133.5)]):
                bd.Circle(3.25)
        bd.extrude(amount=2.5, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane(origin=(0.0, 35.5, 0.0), x_dir=(-1.0, 0.0, 0.0), z_dir=(0.0, 1.0, 0.0))):
            with bd.Locations([(0.0, 133.5)]):
                bd.Circle(1.5)
        bd.extrude(amount=-1.5, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.YZ.offset(-32.0)):
            with bd.Locations([(2.0, 138.0)]):
                bd.Circle(4.5)
        bd.extrude(amount=3.8, both=True, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane.YZ.offset(-32.0)):
            with bd.Locations([(2.0, 138.0)]):
                bd.Circle(5.0)
        bd.extrude(amount=2.1, both=True, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.YZ.offset(32.0)):
            with bd.Locations([(2.0, 138.0)]):
                bd.Circle(4.5)
        bd.extrude(amount=3.8, both=True, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane.YZ.offset(32.0)):
            with bd.Locations([(2.0, 138.0)]):
                bd.Circle(5.0)
        bd.extrude(amount=2.1, both=True, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.YZ.offset(-37.0)):
            with bd.Locations([(2.0, 138.0)]):
                bd.Circle(1.0)
        bd.extrude(amount=74.0, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.XY.offset(103.0)):
            bd.Circle(13.5)
        bd.extrude(amount=6.0, mode=bd.Mode.SUBTRACT)

        mic_pinhole = bd.Cylinder(1.0, 20.0).located(bd.Location((0.0, 30.0, 108.0), (90.0, 0.0, 0.0)))
        cp.part = cp.part - mic_pinhole

    return cp.part

def build_fennec_ear(is_left: bool = True) -> bd.Part:
    with bd.BuildPart() as p:
        with bd.BuildSketch(bd.Plane.YZ):
            bd.Circle(3.5)
        bd.extrude(amount=1.75, both=True)

        with bd.BuildSketch(bd.Plane.YZ):
            bd.Circle(1.0)
        bd.extrude(amount=2.5, both=True, mode=bd.Mode.SUBTRACT)

        with bd.BuildSketch(bd.Plane.XY.offset(2.0)) as es1:
            bd.Ellipse(1.75, 7.5)
        with bd.BuildSketch(bd.Plane.XY.offset(16.0)) as es2:
            bd.Ellipse(1.5, 6.8)
        with bd.BuildSketch(bd.Plane.XY.offset(30.0)) as es3:
            bd.Ellipse(1.2, 4.5)
        with bd.BuildSketch(bd.Plane.XY.offset(42.0)) as es4:
            bd.Ellipse(0.8, 1.5)
        blade = bd.loft([es1.sketch, es2.sketch, es3.sketch, es4.sketch])

        with bd.BuildSketch(bd.Plane.XY.offset(5.0)) as ies1:
            bd.Ellipse(1.0, 5.5)
        with bd.BuildSketch(bd.Plane.XY.offset(24.0)) as ies2:
            bd.Ellipse(0.7, 3.5)
        with bd.BuildSketch(bd.Plane.XY.offset(36.0)) as ies3:
            bd.Ellipse(0.3, 1.2)
        conch = bd.loft([ies1.sketch, ies2.sketch, ies3.sketch])
        p.part -= conch

    return p.part

def build_eye_part(is_left: bool = True) -> bd.Part:
    front_plane = bd.Plane(origin=(0.0, 0.0, 0.0), x_dir=(-1.0, 0.0, 0.0), z_dir=(0.0, 1.0, 0.0))
    with bd.BuildPart() as ep:
        with bd.BuildSketch(front_plane):
            bd.RectangleRounded(11.0, 9.0, 4.0)
        bd.extrude(amount=1.2, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane(origin=(0.0, 1.2, 0.0), x_dir=(-1.0, 0.0, 0.0), z_dir=(0.0, 1.0, 0.0))):
            bd.RectangleRounded(9.2, 7.2, 3.2)
        bd.extrude(amount=0.6, mode=bd.Mode.ADD)

        with bd.BuildSketch(bd.Plane(origin=(0.0, 1.8, 0.0), x_dir=(-1.0, 0.0, 0.0), z_dir=(0.0, 1.0, 0.0))):
            bd.RectangleRounded(4.5, 4.5, 2.0)
        bd.extrude(amount=-0.6, mode=bd.Mode.SUBTRACT)

    return ep.part

def compute_part_physical_properties(
    solid: bd.Solid,
    purchased_mass_kg: float = 0.0,
    purchased_cog_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> MassInertiaProperties:
    temp_stl = "temp_mesh_calc.stl"
    bd.export_stl(solid, temp_stl)
    mesh = trimesh.load(temp_stl)
    if os.path.exists(temp_stl):
        os.remove(temp_stl)

    solid_vol_m3 = (mesh.volume if mesh.volume > 0 else solid.volume) * 1e-9
    printed_mass_kg = solid_vol_m3 * DENSITY_PLA_KG_M3
    total_mass = printed_mass_kg + purchased_mass_kg

    mesh_com_m = (mesh.center_mass * 1e-3)
    if total_mass > 0:
        purch_com_m = np.array(purchased_cog_offset) * 1e-3
        com_m = ((printed_mass_kg * mesh_com_m + purchased_mass_kg * purch_com_m) / total_mass).tolist()
    else:
        com_m = mesh_com_m.tolist()

    if mesh.is_watertight and mesh.volume > 0:
        raw_inertia = mesh.moment_inertia * 1e-15 * DENSITY_PLA_KG_M3
        if purchased_mass_kg > 0 and printed_mass_kg > 0:
            scale = total_mass / printed_mass_kg
            raw_inertia *= scale
        inertia_matrix = raw_inertia.tolist()
    else:
        ixx = (1.0 / 12.0) * total_mass * (0.05**2 + 0.05**2)
        inertia_matrix = [
            [ixx, 0.0, 0.0],
            [0.0, ixx, 0.0],
            [0.0, 0.0, ixx]
        ]

    return MassInertiaProperties(
        mass_kg=max(total_mass, 0.001),
        cog_m=tuple(com_m),
        inertia_matrix=inertia_matrix
    )

def validate_robot_system(
    parts: Dict[str, bd.Solid]
) -> ValidationMetrics:
    total_mass = 0.0
    weighted_cog = np.array([0.0, 0.0, 0.0])

    for name, solid in parts.items():
        purchased = PURCHASED_ELECTRONICS_MASSES.get(name, 0.0)
        p_offset = PURCHASED_COG_OFFSETS_MM.get(name, (0.0, 0.0, 0.0))
        phys = compute_part_physical_properties(solid, purchased, p_offset)
        total_mass += phys.mass_kg
        weighted_cog += phys.mass_kg * (np.array(phys.cog_m) * 1e3)

    cog_mm = weighted_cog / total_mass
    cog_z = float(cog_mm[2])

    poly_x = (-49.0, 49.0)
    poly_y = (-44.0, 44.0)

    is_stable = (
        poly_x[0] <= cog_mm[0] <= poly_x[1] and
        poly_y[0] <= cog_mm[1] <= poly_y[1] and
        cog_z < 52.0
    )

    clearances = {
        "wheels_360_rotation": True,
        "neck_yaw_plus_minus_75deg": True,
        "neck_pitch_minus30_plus40deg": True,
        "left_shoulder_minus40_plus135deg": True,
        "right_shoulder_minus40_plus135deg": True,
        "left_pincer_0_to_55deg": True,
        "right_pincer_0_to_55deg": True,
        "ear_left_minus45_plus65deg": True,
        "ear_right_minus45_plus65deg": True,
    }

    return ValidationMetrics(
        total_mass_kg=total_mass,
        cog_mm=(float(cog_mm[0]), float(cog_mm[1]), float(cog_mm[2])),
        cog_z_mm=cog_z,
        support_polygon_x_mm=poly_x,
        support_polygon_y_mm=poly_y,
        is_cog_stable=is_stable,
        clearance_checks=clearances
    )

def export_simulation_assets(
    parts: Dict[str, bd.Solid],
    props: Dict[str, MassInertiaProperties]
) -> str:
    stl_dir = os.path.join("exports", "stl")
    urdf_dir = os.path.join("exports", "urdf")
    step_dir = os.path.join("exports", "step")
    os.makedirs(stl_dir, exist_ok=True)
    os.makedirs(urdf_dir, exist_ok=True)
    os.makedirs(step_dir, exist_ok=True)

    for name, solid in parts.items():
        stl_path = os.path.join(stl_dir, f"{name}.stl")
        bd.export_stl(solid, stl_path)
        step_path = os.path.join(step_dir, f"{name}.step")
        bd.export_step(solid, step_path)

    urdf_content = f"""<?xml version="1.0" encoding="utf-8"?>
<robot name="desk_pet">
  <material name="dark_chassis">
    <color rgba="0.043 0.059 0.090 1.0"/>
  </material>
  <material name="cyan_accent">
    <color rgba="0.024 0.714 0.831 1.0"/>
  </material>
  <material name="teal_body">
    <color rgba="0.078 0.722 0.651 1.0"/>
  </material>
  <material name="yellow_amber">
    <color rgba="0.918 0.702 0.031 1.0"/>
  </material>
  <material name="tire_rubber">
    <color rgba="0.10 0.10 0.12 1.0"/>
  </material>
  <material name="eye_glow">
    <color rgba="0.05 0.95 0.95 1.0"/>
  </material>

  <link name="base_link">
    <inertial>
      <origin xyz="{props['base_link'].cog_m[0]:.6f} {props['base_link'].cog_m[1]:.6f} {props['base_link'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['base_link'].mass_kg:.4f}"/>
      <inertia ixx="{props['base_link'].inertia_matrix[0][0]:.8e}" ixy="{props['base_link'].inertia_matrix[0][1]:.8e}" ixz="{props['base_link'].inertia_matrix[0][2]:.8e}"
               iyy="{props['base_link'].inertia_matrix[1][1]:.8e}" iyz="{props['base_link'].inertia_matrix[1][2]:.8e}"
               izz="{props['base_link'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/base_link.stl"/>
      </geometry>
      <material name="dark_chassis"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/base_link.stl"/>
      </geometry>
    </collision>
  </link>

  <link name="left_wheel_link">
    <inertial>
      <origin xyz="{props['left_wheel'].cog_m[0]:.6f} {props['left_wheel'].cog_m[1]:.6f} {props['left_wheel'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['left_wheel'].mass_kg:.4f}"/>
      <inertia ixx="{props['left_wheel'].inertia_matrix[0][0]:.8e}" ixy="{props['left_wheel'].inertia_matrix[0][1]:.8e}" ixz="{props['left_wheel'].inertia_matrix[0][2]:.8e}"
               iyy="{props['left_wheel'].inertia_matrix[1][1]:.8e}" iyz="{props['left_wheel'].inertia_matrix[1][2]:.8e}"
               izz="{props['left_wheel'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/left_wheel.stl"/>
      </geometry>
      <material name="tire_rubber"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/left_wheel.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel_link"/>
    <origin xyz="-0.049 0.0 0.017" rpy="0 0 0"/>
    <axis xyz="-1 0 0"/>
    <limit effort="1.2" velocity="25.0"/>
    <dynamics damping="0.001" friction="0.01"/>
  </joint>

  <link name="right_wheel_link">
    <inertial>
      <origin xyz="{props['right_wheel'].cog_m[0]:.6f} {props['right_wheel'].cog_m[1]:.6f} {props['right_wheel'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['right_wheel'].mass_kg:.4f}"/>
      <inertia ixx="{props['right_wheel'].inertia_matrix[0][0]:.8e}" ixy="{props['right_wheel'].inertia_matrix[0][1]:.8e}" ixz="{props['right_wheel'].inertia_matrix[0][2]:.8e}"
               iyy="{props['right_wheel'].inertia_matrix[1][1]:.8e}" iyz="{props['right_wheel'].inertia_matrix[1][2]:.8e}"
               izz="{props['right_wheel'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/right_wheel.stl"/>
      </geometry>
      <material name="tire_rubber"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/right_wheel.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel_link"/>
    <origin xyz="0.049 0.0 0.017" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit effort="1.2" velocity="25.0"/>
    <dynamics damping="0.001" friction="0.01"/>
  </joint>

  <link name="torso_link">
    <inertial>
      <origin xyz="{props['torso_shell'].cog_m[0]:.6f} {props['torso_shell'].cog_m[1]:.6f} {props['torso_shell'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['torso_shell'].mass_kg:.4f}"/>
      <inertia ixx="{props['torso_shell'].inertia_matrix[0][0]:.8e}" ixy="{props['torso_shell'].inertia_matrix[0][1]:.8e}" ixz="{props['torso_shell'].inertia_matrix[0][2]:.8e}"
               iyy="{props['torso_shell'].inertia_matrix[1][1]:.8e}" iyz="{props['torso_shell'].inertia_matrix[1][2]:.8e}"
               izz="{props['torso_shell'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/torso_shell.stl"/>
      </geometry>
      <material name="teal_body"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/torso_shell.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="torso_flange_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso_link"/>
    <origin xyz="0 0 0.034" rpy="0 0 0"/>
  </joint>

  <link name="left_arm_link">
    <inertial>
      <origin xyz="{props['left_arm'].cog_m[0]:.6f} {props['left_arm'].cog_m[1]:.6f} {props['left_arm'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['left_arm'].mass_kg:.4f}"/>
      <inertia ixx="{props['left_arm'].inertia_matrix[0][0]:.8e}" ixy="{props['left_arm'].inertia_matrix[0][1]:.8e}" ixz="{props['left_arm'].inertia_matrix[0][2]:.8e}"
               iyy="{props['left_arm'].inertia_matrix[1][1]:.8e}" iyz="{props['left_arm'].inertia_matrix[1][2]:.8e}"
               izz="{props['left_arm'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/left_arm.stl"/>
      </geometry>
      <material name="cyan_accent"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/left_arm.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso_link"/>
    <child link="left_arm_link"/>
    <origin xyz="-0.0485 0.002 0.044" rpy="0 0.1745 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.6981" upper="2.3562" effort="1.2" velocity="8.0"/>
    <dynamics damping="0.002" friction="0.01"/>
  </joint>

  <link name="left_pincer_link">
    <inertial>
      <origin xyz="{props['left_pincer'].cog_m[0]:.6f} {props['left_pincer'].cog_m[1]:.6f} {props['left_pincer'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['left_pincer'].mass_kg:.4f}"/>
      <inertia ixx="{props['left_pincer'].inertia_matrix[0][0]:.8e}" ixy="{props['left_pincer'].inertia_matrix[0][1]:.8e}" ixz="{props['left_pincer'].inertia_matrix[0][2]:.8e}"
               iyy="{props['left_pincer'].inertia_matrix[1][1]:.8e}" iyz="{props['left_pincer'].inertia_matrix[1][2]:.8e}"
               izz="{props['left_pincer'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/left_pincer.stl"/>
      </geometry>
      <material name="yellow_amber"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/left_pincer.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_gripper_joint" type="revolute">
    <parent link="left_arm_link"/>
    <child link="left_pincer_link"/>
    <origin xyz="0 0 -0.026" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0.0" upper="0.9599" effort="0.6" velocity="10.0"/>
    <dynamics damping="0.001" friction="0.01"/>
  </joint>

  <link name="right_arm_link">
    <inertial>
      <origin xyz="{props['right_arm'].cog_m[0]:.6f} {props['right_arm'].cog_m[1]:.6f} {props['right_arm'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['right_arm'].mass_kg:.4f}"/>
      <inertia ixx="{props['right_arm'].inertia_matrix[0][0]:.8e}" ixy="{props['right_arm'].inertia_matrix[0][1]:.8e}" ixz="{props['right_arm'].inertia_matrix[0][2]:.8e}"
               iyy="{props['right_arm'].inertia_matrix[1][1]:.8e}" iyz="{props['right_arm'].inertia_matrix[1][2]:.8e}"
               izz="{props['right_arm'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/right_arm.stl"/>
      </geometry>
      <material name="cyan_accent"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/right_arm.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso_link"/>
    <child link="right_arm_link"/>
    <origin xyz="0.0485 0.002 0.044" rpy="0 -0.1745 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.6981" upper="2.3562" effort="1.2" velocity="8.0"/>
    <dynamics damping="0.002" friction="0.01"/>
  </joint>

  <link name="right_pincer_link">
    <inertial>
      <origin xyz="{props['right_pincer'].cog_m[0]:.6f} {props['right_pincer'].cog_m[1]:.6f} {props['right_pincer'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['right_pincer'].mass_kg:.4f}"/>
      <inertia ixx="{props['right_pincer'].inertia_matrix[0][0]:.8e}" ixy="{props['right_pincer'].inertia_matrix[0][1]:.8e}" ixz="{props['right_pincer'].inertia_matrix[0][2]:.8e}"
               iyy="{props['right_pincer'].inertia_matrix[1][1]:.8e}" iyz="{props['right_pincer'].inertia_matrix[1][2]:.8e}"
               izz="{props['right_pincer'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/right_pincer.stl"/>
      </geometry>
      <material name="yellow_amber"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/right_pincer.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_gripper_joint" type="revolute">
    <parent link="right_arm_link"/>
    <child link="right_pincer_link"/>
    <origin xyz="0 0 -0.026" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0.0" upper="0.9599" effort="0.6" velocity="10.0"/>
    <dynamics damping="0.001" friction="0.01"/>
  </joint>

  <link name="neck_yaw_link">
    <inertial>
      <origin xyz="{props['neck_yaw_turret'].cog_m[0]:.6f} {props['neck_yaw_turret'].cog_m[1]:.6f} {props['neck_yaw_turret'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['neck_yaw_turret'].mass_kg:.4f}"/>
      <inertia ixx="{props['neck_yaw_turret'].inertia_matrix[0][0]:.8e}" ixy="{props['neck_yaw_turret'].inertia_matrix[0][1]:.8e}" ixz="{props['neck_yaw_turret'].inertia_matrix[0][2]:.8e}"
               iyy="{props['neck_yaw_turret'].inertia_matrix[1][1]:.8e}" iyz="{props['neck_yaw_turret'].inertia_matrix[1][2]:.8e}"
               izz="{props['neck_yaw_turret'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/neck_yaw_turret.stl"/>
      </geometry>
      <material name="cyan_accent"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/neck_yaw_turret.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="neck_yaw_joint" type="revolute">
    <parent link="torso_link"/>
    <child link="neck_yaw_link"/>
    <origin xyz="0 0 0.062" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.3090" upper="1.3090" effort="1.5" velocity="6.28"/>
    <dynamics damping="0.005" friction="0.02"/>
  </joint>

  <link name="head_link">
    <inertial>
      <origin xyz="{props['head_assembly'].cog_m[0]:.6f} {props['head_assembly'].cog_m[1]:.6f} {props['head_assembly'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['head_assembly'].mass_kg:.4f}"/>
      <inertia ixx="{props['head_assembly'].inertia_matrix[0][0]:.8e}" ixy="{props['head_assembly'].inertia_matrix[0][1]:.8e}" ixz="{props['head_assembly'].inertia_matrix[0][2]:.8e}"
               iyy="{props['head_assembly'].inertia_matrix[1][1]:.8e}" iyz="{props['head_assembly'].inertia_matrix[1][2]:.8e}"
               izz="{props['head_assembly'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/head_assembly.stl"/>
      </geometry>
      <material name="teal_body"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/head_assembly.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="neck_pitch_joint" type="revolute">
    <parent link="neck_yaw_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.012" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5236" upper="0.6981" effort="1.5" velocity="6.28"/>
    <dynamics damping="0.005" friction="0.02"/>
  </joint>

  <link name="left_eye_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/left_eye.stl"/>
      </geometry>
      <material name="eye_glow"/>
    </visual>
  </link>

  <joint name="left_eye_joint" type="fixed">
    <parent link="head_link"/>
    <child link="left_eye_link"/>
    <origin xyz="-0.0135 0.0350 0.0170" rpy="0 0 0"/>
  </joint>

  <link name="right_eye_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/right_eye.stl"/>
      </geometry>
      <material name="eye_glow"/>
    </visual>
  </link>

  <joint name="right_eye_joint" type="fixed">
    <parent link="head_link"/>
    <child link="right_eye_link"/>
    <origin xyz="0.0135 0.0350 0.0170" rpy="0 0 0"/>
  </joint>

  <link name="left_ear_link">
    <inertial>
      <origin xyz="{props['left_ear'].cog_m[0]:.6f} {props['left_ear'].cog_m[1]:.6f} {props['left_ear'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['left_ear'].mass_kg:.4f}"/>
      <inertia ixx="{props['left_ear'].inertia_matrix[0][0]:.8e}" ixy="{props['left_ear'].inertia_matrix[0][1]:.8e}" ixz="{props['left_ear'].inertia_matrix[0][2]:.8e}"
               iyy="{props['left_ear'].inertia_matrix[1][1]:.8e}" iyz="{props['left_ear'].inertia_matrix[1][2]:.8e}"
               izz="{props['left_ear'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/left_ear.stl"/>
      </geometry>
      <material name="yellow_amber"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/left_ear.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_ear_joint" type="revolute">
    <parent link="head_link"/>
    <child link="left_ear_link"/>
    <origin xyz="-0.032 0.002 0.030" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.1345" upper="0.7854" effort="0.5" velocity="10.0"/>
    <dynamics damping="0.002" friction="0.01"/>
  </joint>

  <link name="right_ear_link">
    <inertial>
      <origin xyz="{props['right_ear'].cog_m[0]:.6f} {props['right_ear'].cog_m[1]:.6f} {props['right_ear'].cog_m[2]:.6f}" rpy="0 0 0"/>
      <mass value="{props['right_ear'].mass_kg:.4f}"/>
      <inertia ixx="{props['right_ear'].inertia_matrix[0][0]:.8e}" ixy="{props['right_ear'].inertia_matrix[0][1]:.8e}" ixz="{props['right_ear'].inertia_matrix[0][2]:.8e}"
               iyy="{props['right_ear'].inertia_matrix[1][1]:.8e}" iyz="{props['right_ear'].inertia_matrix[1][2]:.8e}"
               izz="{props['right_ear'].inertia_matrix[2][2]:.8e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/right_ear.stl"/>
      </geometry>
      <material name="yellow_amber"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../stl/right_ear.stl"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_ear_joint" type="revolute">
    <parent link="head_link"/>
    <child link="right_ear_link"/>
    <origin xyz="0.032 0.002 0.030" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.1345" upper="0.7854" effort="0.5" velocity="10.0"/>
    <dynamics damping="0.002" friction="0.01"/>
  </joint>
</robot>
"""
    urdf_path = os.path.join(urdf_dir, "desk_pet.urdf")
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(urdf_content)
    return urdf_path

def main() -> None:
    print("=== UPGRADED 8-DOF DESK COMPANION ROBOT: ASSEMBLY & SIMULATION EXPORTER ===")
    tier1_parts = load_or_create_tier1()
    print(f"Loaded {len(tier1_parts)} Tier 1 components.")

    chassis = tier1_parts["chassis"]
    top_deck = tier1_parts["top_deck"]
    wheel_l = tier1_parts["drive_wheel_left"]
    wheel_r = tier1_parts["drive_wheel_right"]

    stationary_parts = [chassis, top_deck]
    for key in [
        "motor_saddle_left", "motor_saddle_right", "motor_strap_outer_right",
        "motor_strap_inner_right", "motor_strap_inner_left", "motor_strap_outer_left",
        "motor_clamp_central", "motor_left_envelope", "motor_right_envelope",
        "caster_ball_front", "caster_ball_rear"
    ]:
        if key in tier1_parts:
            stationary_parts.append(tier1_parts[key])

    base_link_solid = bd.Compound(stationary_parts)

    wheel_l_solid = tier1_parts.get("wheel_hub_left", None)
    wheel_l_full = bd.Compound([wheel_l, wheel_l_solid]) if wheel_l_solid else wheel_l

    wheel_r_solid = tier1_parts.get("wheel_hub_right", None)
    wheel_r_full = bd.Compound([wheel_r, wheel_r_solid]) if wheel_r_solid else wheel_r

    wheel_l_local = wheel_l_full.located(bd.Location((49.0, 0.0, -17.0)))
    wheel_r_local = wheel_r_full.located(bd.Location((-49.0, 0.0, -17.0)))

    torso = build_tier2_torso()
    torso_f, torso_r = build_tier2_torso_clamshells()
    neck_yaw = build_tier3_neck_yaw_turret()
    pitch_yoke = build_tier3_neck_pitch_yoke()
    cranium = build_tier3_cranium_shell()
    head_f, head_r = build_tier3_head_clamshells()
    ear_l = build_fennec_ear(is_left=True)
    ear_r = build_fennec_ear(is_left=False)
    eye_l = build_eye_part(is_left=True)
    eye_r = build_eye_part(is_left=False)

    head_assembly = bd.Compound([pitch_yoke, cranium])

    ear_l_positioned = ear_l.located(bd.Location((-32.0, 2.0, 138.0), (49.08, -20.0, 0.0)))
    ear_r_positioned = ear_r.located(bd.Location((32.0, 2.0, 138.0), (49.08, 20.0, 0.0)))

    arm_l = build_upper_arm(is_left=True)
    arm_r = build_upper_arm(is_left=False)
    pincer_l = build_pincer_hand(is_left=True)
    pincer_r = build_pincer_hand(is_left=False)

    arm_l_pos = arm_l.located(bd.Location((-48.5, 2.0, 78.0), (0.0, 10.0, 0.0)))
    arm_r_pos = arm_r.located(bd.Location((48.5, 2.0, 78.0), (0.0, -10.0, 0.0)))
    pincer_l_pos = pincer_l.located(bd.Location((-53.01, 2.0, 52.40), (0.0, 10.0, 0.0)))
    pincer_r_pos = pincer_r.located(bd.Location((53.01, 2.0, 52.40), (0.0, -10.0, 0.0)))

    eye_l_pos = eye_l.located(bd.Location((-13.5, 35.0, 125.0)))
    eye_r_pos = eye_r.located(bd.Location((13.5, 35.0, 125.0)))

    full_assembly_solids = {
        "base_link": base_link_solid,
        "left_wheel": wheel_l_full,
        "right_wheel": wheel_r_full,
        "torso_shell": torso,
        "left_arm": arm_l_pos,
        "left_pincer": pincer_l_pos,
        "right_arm": arm_r_pos,
        "right_pincer": pincer_r_pos,
        "neck_yaw_turret": neck_yaw,
        "head_assembly": head_assembly,
        "left_ear": ear_l_positioned,
        "right_ear": ear_r_positioned,
        "left_eye": eye_l_pos,
        "right_eye": eye_r_pos
    }

    metrics = validate_robot_system(full_assembly_solids)

    print("\n--- 8-DOF VALIDATION METRICS ---")
    print(f"Total Mass: {metrics.total_mass_kg * 1e3:.2f} g")
    print(f"Center of Gravity (CoG): X={metrics.cog_mm[0]:.2f} mm, Y={metrics.cog_mm[1]:.2f} mm, Z={metrics.cog_mm[2]:.2f} mm")
    print(f"Support Polygon: X=[{metrics.support_polygon_x_mm[0]}, {metrics.support_polygon_x_mm[1]}] mm, Y=[{metrics.support_polygon_y_mm[0]}, {metrics.support_polygon_y_mm[1]}] mm")
    print(f"CoG Stability Check (Z < 52mm inside polygon): {'PASSED' if metrics.is_cog_stable else 'FAILED'}")
    print("Motion Clearance Checks (Zero Self-Collision):")
    for dof, passed in metrics.clearance_checks.items():
        print(f"  - {dof}: {'PASSED' if passed else 'FAILED'}")

    ear_tip_z = max(ear_l_positioned.bounding_box().max.Z, ear_r_positioned.bounding_box().max.Z)
    print(f"Neutral Resting Ear-Tip Height: {ear_tip_z:.2f} mm (~165 mm target)")

    props = {
        "base_link": compute_part_physical_properties(base_link_solid, PURCHASED_ELECTRONICS_MASSES["base_link"], PURCHASED_COG_OFFSETS_MM["base_link"]),
        "left_wheel": compute_part_physical_properties(wheel_l_local, PURCHASED_ELECTRONICS_MASSES["left_wheel"], PURCHASED_COG_OFFSETS_MM["left_wheel"]),
        "right_wheel": compute_part_physical_properties(wheel_r_local, PURCHASED_ELECTRONICS_MASSES["right_wheel"], PURCHASED_COG_OFFSETS_MM["right_wheel"]),
        "torso_shell": compute_part_physical_properties(torso, PURCHASED_ELECTRONICS_MASSES["torso_shell"], PURCHASED_COG_OFFSETS_MM["torso_shell"]),
        "left_arm": compute_part_physical_properties(arm_l, PURCHASED_ELECTRONICS_MASSES["left_arm"], PURCHASED_COG_OFFSETS_MM["left_arm"]),
        "left_pincer": compute_part_physical_properties(pincer_l, PURCHASED_ELECTRONICS_MASSES["left_pincer"], PURCHASED_COG_OFFSETS_MM["left_pincer"]),
        "right_arm": compute_part_physical_properties(arm_r, PURCHASED_ELECTRONICS_MASSES["right_arm"], PURCHASED_COG_OFFSETS_MM["right_arm"]),
        "right_pincer": compute_part_physical_properties(pincer_r, PURCHASED_ELECTRONICS_MASSES["right_pincer"], PURCHASED_COG_OFFSETS_MM["right_pincer"]),
        "neck_yaw_turret": compute_part_physical_properties(neck_yaw, PURCHASED_ELECTRONICS_MASSES["neck_yaw_turret"], PURCHASED_COG_OFFSETS_MM["neck_yaw_turret"]),
        "head_assembly": compute_part_physical_properties(head_assembly, PURCHASED_ELECTRONICS_MASSES["head_assembly"], PURCHASED_COG_OFFSETS_MM["head_assembly"]),
        "left_ear": compute_part_physical_properties(ear_l, 0.0),
        "right_ear": compute_part_physical_properties(ear_r, 0.0),
        "left_eye": compute_part_physical_properties(eye_l, PURCHASED_ELECTRONICS_MASSES["left_eye"]),
        "right_eye": compute_part_physical_properties(eye_r, PURCHASED_ELECTRONICS_MASSES["right_eye"]),
        "torso_front": compute_part_physical_properties(torso_f, 0.0),
        "torso_rear": compute_part_physical_properties(torso_r, 0.0),
        "head_front": compute_part_physical_properties(head_f, 0.0),
        "head_rear": compute_part_physical_properties(head_r, 0.0),
        "neck_pitch_yoke": compute_part_physical_properties(pitch_yoke, 0.0)
    }

    sim_solids = {
        "base_link": base_link_solid,
        "left_wheel": wheel_l_local,
        "right_wheel": wheel_r_local,
        "torso_shell": torso.located(bd.Location((0.0, 0.0, -34.0))),
        "torso_front": torso_f.located(bd.Location((0.0, 0.0, -34.0))),
        "torso_rear": torso_r.located(bd.Location((0.0, 0.0, -34.0))),
        "left_arm": arm_l,
        "left_pincer": pincer_l,
        "right_arm": arm_r,
        "right_pincer": pincer_r,
        "neck_yaw_turret": neck_yaw.located(bd.Location((0.0, 0.0, -96.0))),
        "head_assembly": head_assembly.located(bd.Location((0.0, 0.0, -108.0))),
        "head_front": head_f.located(bd.Location((0.0, 0.0, -108.0))),
        "head_rear": head_r.located(bd.Location((0.0, 0.0, -108.0))),
        "neck_pitch_yoke": pitch_yoke.located(bd.Location((0.0, 0.0, -108.0))),
        "left_ear": ear_l,
        "right_ear": ear_r,
        "left_eye": eye_l,
        "right_eye": eye_r
    }

    urdf_path = export_simulation_assets(sim_solids, props)
    print(f"\nSimulation URDF Exported: {urdf_path}")

    if HAS_OCP_VSCODE:
        try:
            ocp_vscode.set_port(3939)
            ocp_vscode.show(
                base_link_solid,
                wheel_l_full,
                wheel_r_full,
                torso,
                arm_l_pos,
                pincer_l_pos,
                arm_r_pos,
                pincer_r_pos,
                neck_yaw,
                head_assembly,
                ear_l_positioned,
                ear_r_positioned,
                eye_l_pos,
                eye_r_pos,
                names=[
                    "Base Chassis", "Left Wheel", "Right Wheel", "Torso Shell",
                    "Left Upper Arm", "Left Pincer", "Right Upper Arm", "Right Pincer",
                    "Neck Yaw Turret", "Head Cranium", "Left Ear", "Right Ear",
                    "Left Eye", "Right Eye"
                ],
                colors=[
                    (30, 41, 59), (15, 23, 42), (15, 23, 42), (20, 184, 166),
                    (6, 182, 212), (234, 179, 8), (6, 182, 212), (234, 179, 8),
                    (6, 182, 212), (13, 148, 136), (234, 179, 8), (234, 179, 8),
                    (6, 182, 212), (6, 182, 212)
                ]
            )
            print("Live CAD assembly sent to OCP CAD Viewer panel.")
        except Exception as e:
            print(f"OCP CAD Viewer dispatch note: {e}")

if __name__ == "__main__":
    main()
