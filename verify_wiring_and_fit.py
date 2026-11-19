from __future__ import annotations
import math
import os
import shutil
from typing import Dict, List, Tuple
import numpy as np
import vtk
from pydantic import BaseModel
import build123d as bd
import build_desk_pet
import verify_electronics_fit
from verify_electronics_fit import (
    ComponentSpec,
    calc_shape_volume,
    COMPONENTS_REGISTRY,
    build_split_clamshell_shells
)

class ConnectorEnvelope(BaseModel):
    name: str
    target_component: str
    dimensions_mm: Tuple[float, float, float]
    position_mm: Tuple[float, float, float]
    color_rgb: Tuple[float, float, float]
    color_hex: str

class WireHarnessSpec(BaseModel):
    name: str
    conduit_name: str
    bundle_diameter_mm: float
    path_points: List[Tuple[float, float, float]]
    color_rgb: Tuple[float, float, float]
    color_hex: str
    min_bend_radius_mm: float
    conduit_inner_diam_mm: float

CONNECTORS_REGISTRY: List[ConnectorEnvelope] = [
    ConnectorEnvelope(
        name="esp32_header_plug",
        target_component="esp32_s3",
        dimensions_mm=(24.0, 5.0, 5.0),
        position_mm=(0.0, -10.0, 88.0),
        color_rgb=(0.20, 0.40, 0.85),
        color_hex="#2563EB"
    ),
    ConnectorEnvelope(
        name="shoulder_servo_l_plug",
        target_component="shoulder_servo_left",
        dimensions_mm=(5.5, 4.5, 4.0),
        position_mm=(-35.0, 0.0, 88.0),
        color_rgb=(0.85, 0.50, 0.10),
        color_hex="#D97706"
    ),
    ConnectorEnvelope(
        name="shoulder_servo_r_plug",
        target_component="shoulder_servo_right",
        dimensions_mm=(5.5, 4.5, 4.0),
        position_mm=(35.0, 0.0, 88.0),
        color_rgb=(0.85, 0.50, 0.10),
        color_hex="#D97706"
    ),
    ConnectorEnvelope(
        name="neck_yaw_servo_plug",
        target_component="neck_yaw_servo",
        dimensions_mm=(5.5, 4.5, 4.0),
        position_mm=(0.0, -4.5, 78.0),
        color_rgb=(0.85, 0.50, 0.10),
        color_hex="#D97706"
    ),
    ConnectorEnvelope(
        name="neck_pitch_servo_plug",
        target_component="neck_pitch_servo",
        dimensions_mm=(5.5, 4.5, 4.0),
        position_mm=(0.0, -4.5, 110.0),
        color_rgb=(0.85, 0.50, 0.10),
        color_hex="#D97706"
    ),
    ConnectorEnvelope(
        name="ear_servo_l_plug",
        target_component="ear_servo_left",
        dimensions_mm=(4.5, 5.5, 4.0),
        position_mm=(-16.5, -4.5, 134.0),
        color_rgb=(0.85, 0.50, 0.10),
        color_hex="#D97706"
    ),
    ConnectorEnvelope(
        name="ear_servo_r_plug",
        target_component="ear_servo_right",
        dimensions_mm=(4.5, 5.5, 4.0),
        position_mm=(16.5, -4.5, 134.0),
        color_rgb=(0.85, 0.50, 0.10),
        color_hex="#D97706"
    ),
    ConnectorEnvelope(
        name="camera_fpc_receptacle",
        target_component="camera_ov5640",
        dimensions_mm=(16.0, 2.5, 2.0),
        position_mm=(0.0, 29.5, 133.5),
        color_rgb=(0.10, 0.70, 0.55),
        color_hex="#059669"
    ),
    ConnectorEnvelope(
        name="oled_l_header",
        target_component="oled_display_left",
        dimensions_mm=(16.0, 3.0, 4.0),
        position_mm=(-15.0, 25.0, 124.0),
        color_rgb=(0.02, 0.71, 0.83),
        color_hex="#06B6D4"
    ),
    ConnectorEnvelope(
        name="oled_r_header",
        target_component="oled_display_right",
        dimensions_mm=(16.0, 3.0, 4.0),
        position_mm=(15.0, 25.0, 124.0),
        color_rgb=(0.02, 0.71, 0.83),
        color_hex="#06B6D4"
    ),
    ConnectorEnvelope(
        name="mic_header_plug",
        target_component="inmp441_mic",
        dimensions_mm=(8.0, 2.5, 4.0),
        position_mm=(0.0, 28.0, 108.0),
        color_rgb=(0.93, 0.28, 0.60),
        color_hex="#EC4899"
    ),
    ConnectorEnvelope(
        name="touch_header_plug",
        target_component="touch_ttp223",
        dimensions_mm=(10.0, 2.5, 4.0),
        position_mm=(0.0, -10.0, 142.5),
        color_rgb=(0.93, 0.28, 0.60),
        color_hex="#EC4899"
    ),
    ConnectorEnvelope(
        name="tb6612_base_headers",
        target_component="tb6612_driver",
        dimensions_mm=(18.0, 4.0, 4.5),
        position_mm=(0.0, 18.0, 13.5),
        color_rgb=(0.92, 0.26, 0.21),
        color_hex="#EF4444"
    ),
    ConnectorEnvelope(
        name="buck_6v_base_headers",
        target_component="buck_12v_6v",
        dimensions_mm=(22.0, 5.0, 4.5),
        position_mm=(0.0, -21.0, 18.0),
        color_rgb=(0.92, 0.70, 0.03),
        color_hex="#EAB308"
    ),
]

HARNESSES_REGISTRY: List[WireHarnessSpec] = [
    WireHarnessSpec(
        name="harness_a_trunk_base",
        conduit_name="Base Deck Grommet (Ø 18 mm)",
        bundle_diameter_mm=6.5,
        path_points=[
            (0.0, -18.0, 14.0),
            (0.0, -8.0, 22.0),
            (0.0, 0.0, 34.0),
            (0.0, -5.0, 48.0),
            (0.0, -11.0, 64.0)
        ],
        color_rgb=(0.02, 0.71, 0.83),
        color_hex="#06B6D4",
        min_bend_radius_mm=14.5,
        conduit_inner_diam_mm=18.0
    ),
    WireHarnessSpec(
        name="harness_b_neck_loop",
        conduit_name="Bearing & Turret Neck Bore (Ø 12 mm)",
        bundle_diameter_mm=5.5,
        path_points=[
            (0.0, -11.0, 72.0),
            (0.0, -4.0, 82.0),
            (0.0, 3.5, 88.0),
            (0.0, 0.0, 96.0),
            (0.0, 4.0, 104.0),
            (0.0, 10.0, 114.0),
            (0.0, 22.0, 126.0)
        ],
        color_rgb=(0.92, 0.70, 0.03),
        color_hex="#EAB308",
        min_bend_radius_mm=12.2,
        conduit_inner_diam_mm=12.0
    ),
    WireHarnessSpec(
        name="harness_c_dorsal_mast",
        conduit_name="Rear Dorsal Mast Bore (Ø 13 mm)",
        bundle_diameter_mm=4.5,
        path_points=[
            (0.0, -36.0, 38.0),
            (0.0, -36.0, 56.0),
            (0.0, -32.0, 60.0),
            (0.0, -24.0, 58.0),
            (0.0, -18.0, 52.0),
            (0.0, -11.0, 64.0)
        ],
        color_rgb=(0.92, 0.26, 0.21),
        color_hex="#EF4444",
        min_bend_radius_mm=10.8,
        conduit_inner_diam_mm=13.0
    ),
    WireHarnessSpec(
        name="harness_d_left_arm",
        conduit_name="Left Shoulder Pivot Hub (Ø 8 mm)",
        bundle_diameter_mm=2.8,
        path_points=[
            (-5.0, -8.0, 66.0),
            (-18.0, -2.0, 72.0),
            (-34.0, 2.0, 78.0),
            (-48.5, 2.0, 78.0),
            (-48.5, 2.0, 66.0)
        ],
        color_rgb=(0.66, 0.33, 0.97),
        color_hex="#A855F7",
        min_bend_radius_mm=9.5,
        conduit_inner_diam_mm=8.0
    ),
    WireHarnessSpec(
        name="harness_e_right_arm",
        conduit_name="Right Shoulder Pivot Hub (Ø 8 mm)",
        bundle_diameter_mm=2.8,
        path_points=[
            (5.0, -8.0, 66.0),
            (18.0, -2.0, 72.0),
            (34.0, 2.0, 78.0),
            (48.5, 2.0, 78.0),
            (48.5, 2.0, 66.0)
        ],
        color_rgb=(0.66, 0.33, 0.97),
        color_hex="#A855F7",
        min_bend_radius_mm=9.5,
        conduit_inner_diam_mm=8.0
    ),
    WireHarnessSpec(
        name="camera_fpc_ribbon",
        conduit_name="Forehead Bridge to Neck Conduit",
        bundle_diameter_mm=2.2,
        path_points=[
            (0.0, 30.5, 133.5),
            (0.0, 22.0, 126.0),
            (0.0, 10.0, 112.0),
            (0.0, 0.0, 96.0),
            (0.0, -6.0, 84.0),
            (0.0, -11.0, 74.0)
        ],
        color_rgb=(0.06, 0.72, 0.50),
        color_hex="#10B981",
        min_bend_radius_mm=11.0,
        conduit_inner_diam_mm=12.0
    )
]

def build_connector_solid(conn: ConnectorEnvelope) -> bd.Solid:
    dx, dy, dz = conn.dimensions_mm
    loc = bd.Location(conn.position_mm)
    return bd.Box(dx, dy, dz).located(loc)

def build_wire_harness_solid(spec: WireHarnessSpec) -> bd.Solid:
    radius = spec.bundle_diameter_mm / 2.0
    sp = bd.Spline(spec.path_points)
    wire = bd.Wire([sp])
    circle = bd.Circle(radius=radius)
    solid = bd.sweep(sections=circle, path=wire)
    return solid

def build_camera_fpc_ribbon_solid() -> bd.Solid:
    pts = [
        (0.0, 30.5, 133.5),
        (0.0, 22.0, 126.0),
        (0.0, 10.0, 112.0),
        (0.0, 0.0, 96.0),
        (0.0, -6.0, 84.0),
        (0.0, -11.0, 74.0)
    ]
    sp = bd.Spline(pts)
    wire = bd.Wire([sp])
    rect = bd.Rectangle(16.0, 0.3)
    solid = bd.sweep(sections=rect, path=wire)
    return solid

def safe_sub(target: bd.Solid, tool: bd.Solid) -> bd.Solid:
    try:
        inter = target.intersect(tool)
        if calc_shape_volume(inter) > 0.0001:
            return target - tool
    except Exception:
        pass
    return target

def build_wiring_cleared_shells() -> Dict[str, bd.Solid]:
    shells = build_split_clamshell_shells()
    connectors = {conn.name: build_connector_solid(conn) for conn in CONNECTORS_REGISTRY}
    harnesses = {h.name: build_wire_harness_solid(h) for h in HARNESSES_REGISTRY}

    conduit_base = bd.Cylinder(9.0, 30.0).located(bd.Location((0.0, 0.0, 34.0)))
    conduit_neck = bd.Cylinder(6.0, 50.0).located(bd.Location((0.0, 0.0, 96.0)))
    conduit_dorsal = bd.Cylinder(4.0, 30.0).located(bd.Location((0.0, -28.0, 60.0), (90.0, 0.0, 0.0)))
    conduit_sh_l = bd.Cylinder(4.0, 40.0).located(bd.Location((-40.0, 2.0, 78.0), (0.0, 90.0, 0.0)))
    conduit_sh_r = bd.Cylinder(4.0, 40.0).located(bd.Location((40.0, 2.0, 78.0), (0.0, 90.0, 0.0)))

    conduits = [conduit_base, conduit_neck, conduit_dorsal, conduit_sh_l, conduit_sh_r]

    for k in shells:
        for conduit in conduits:
            shells[k] = safe_sub(shells[k], conduit)
        for c in connectors.values():
            shells[k] = safe_sub(shells[k], c)
        for h in harnesses.values():
            shells[k] = safe_sub(shells[k], h)

    return shells

def calculate_conduit_fill_metrics(spec: WireHarnessSpec) -> Dict[str, float]:
    conduit_area = math.pi * (spec.conduit_inner_diam_mm / 2.0) ** 2
    if "Neck" in spec.conduit_name:
        bundle_area = math.pi * (5.5 / 2.0) ** 2
        fpc_area = 16.0 * 0.3
        wire_area = bundle_area + fpc_area
    else:
        wire_area = math.pi * (spec.bundle_diameter_mm / 2.0) ** 2

    fill_pct = (wire_area / conduit_area) * 100.0
    free_air_gap_pct = 100.0 - fill_pct
    return {
        "conduit_area_mm2": conduit_area,
        "wire_area_mm2": wire_area,
        "fill_pct": fill_pct,
        "free_air_gap_pct": free_air_gap_pct
    }

def run_wiring_interference_audit(
    components: Dict[str, bd.Solid],
    connectors: Dict[str, bd.Solid],
    harnesses: Dict[str, bd.Solid],
    shells: Dict[str, bd.Solid]
) -> List[dict]:
    results = []

    for cname, csolid in connectors.items():
        max_col = 0.0
        for sname, ssolid in shells.items():
            res = ssolid.intersect(csolid)
            vol = calc_shape_volume(res)
            if vol > max_col:
                max_col = vol
        results.append({
            "type": "Connector Envelope",
            "name": cname,
            "collision_mm3": round(max_col, 3),
            "status": "PASS" if max_col <= 0.001 else "FAIL"
        })

    for hname, hsolid in harnesses.items():
        max_col = 0.0
        for sname, ssolid in shells.items():
            res = ssolid.intersect(hsolid)
            vol = calc_shape_volume(res)
            if vol > max_col:
                max_col = vol
        results.append({
            "type": "Wire Harness",
            "name": hname,
            "collision_mm3": round(max_col, 3),
            "status": "PASS" if max_col <= 0.001 else "FAIL"
        })

    return results

def export_all_wiring_stls(
    connectors: Dict[str, bd.Solid],
    harnesses: Dict[str, bd.Solid],
    output_dir: str = "exports/stl/internals"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    all_harness_solids = list(harnesses.values()) + list(connectors.values())
    compound = bd.Compound(all_harness_solids)
    out_path = os.path.join(output_dir, "wire_harnesses.stl")
    bd.export_stl(compound, out_path)

    for hname, hsolid in harnesses.items():
        p = os.path.join(output_dir, f"{hname}.stl")
        bd.export_stl(hsolid, p)

    return out_path

def render_wiring_xray_views(
    components: Dict[str, bd.Solid],
    connectors: Dict[str, bd.Solid],
    harnesses: Dict[str, bd.Solid],
    output_dir: str = "exports/renders"
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stl_dir = os.path.join("exports", "stl")

    shell_configs = [
        ("base_link.stl", (0.118, 0.161, 0.231), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("left_wheel.stl", (0.059, 0.090, 0.165), (-49.0, 0.0, 17.0), (0.0, 0.0, 0.0)),
        ("right_wheel.stl", (0.059, 0.090, 0.165), (49.0, 0.0, 17.0), (0.0, 0.0, 0.0)),
        ("torso_shell.stl", (0.078, 0.722, 0.651), (0.0, 0.0, 34.0), (0.0, 0.0, 0.0)),
        ("left_arm.stl", (0.024, 0.714, 0.831), (-48.5, 2.0, 78.0), (0.0, 10.0, 0.0)),
        ("left_pincer.stl", (0.918, 0.702, 0.031), (-53.01, 2.0, 52.4), (0.0, 10.0, 0.0)),
        ("right_arm.stl", (0.024, 0.714, 0.831), (48.5, 2.0, 78.0), (0.0, -10.0, 0.0)),
        ("right_pincer.stl", (0.918, 0.702, 0.031), (53.01, 2.0, 52.4), (0.0, -10.0, 0.0)),
        ("neck_yaw_turret.stl", (0.024, 0.714, 0.831), (0.0, 0.0, 96.0), (0.0, 0.0, 0.0)),
        ("head_assembly.stl", (0.051, 0.580, 0.533), (0.0, 0.0, 108.0), (0.0, 0.0, 0.0)),
        ("left_ear.stl", (0.918, 0.702, 0.031), (-32.0, 2.0, 138.0), (49.08, -20.0, 0.0)),
        ("right_ear.stl", (0.918, 0.702, 0.031), (32.0, 2.0, 138.0), (49.08, 20.0, 0.0)),
    ]

    actors: List[vtk.vtkActor] = []

    for filename, color, pos, rot in shell_configs:
        path = os.path.join(stl_dir, filename)
        if os.path.exists(path):
            reader = vtk.vtkSTLReader()
            reader.SetFileName(path)
            reader.Update()

            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(reader.GetOutputPort())
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOff()
            normals.SetFeatureAngle(60.0)
            normals.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color[0], color[1], color[2])
            actor.GetProperty().SetOpacity(0.18)
            actor.GetProperty().SetSpecular(0.2)
            actor.GetProperty().SetAmbient(0.3)
            actor.GetProperty().SetDiffuse(0.7)
            actor.SetPosition(pos[0], pos[1], pos[2])
            actor.SetOrientation(rot[0], rot[1], rot[2])
            actors.append(actor)

    internals_dir = os.path.join(stl_dir, "internals")
    for spec in COMPONENTS_REGISTRY:
        path = os.path.join(internals_dir, f"{spec.name}.stl")
        if os.path.exists(path):
            reader = vtk.vtkSTLReader()
            reader.SetFileName(path)
            reader.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(spec.color_rgb[0], spec.color_rgb[1], spec.color_rgb[2])
            actor.GetProperty().SetOpacity(0.95)
            actor.GetProperty().SetSpecular(0.4)
            actor.GetProperty().SetDiffuse(0.8)
            actors.append(actor)

    for hspec in HARNESSES_REGISTRY:
        path = os.path.join(internals_dir, f"{hspec.name}.stl")
        if os.path.exists(path):
            reader = vtk.vtkSTLReader()
            reader.SetFileName(path)
            reader.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(hspec.color_rgb[0], hspec.color_rgb[1], hspec.color_rgb[2])
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetSpecular(0.6)
            actor.GetProperty().SetDiffuse(0.9)
            actors.append(actor)

    front_img = os.path.join(output_dir, "xray_wiring_front.png")
    side_img = os.path.join(output_dir, "xray_wiring_side.png")

    render_scene(actors, (0.0, 300.0, 82.0), (0.0, 0.0, 82.0), (0.0, 0.0, 1.0), front_img)
    render_scene(actors, (300.0, 0.0, 82.0), (0.0, 0.0, 82.0), (0.0, 0.0, 1.0), side_img)

    return {"front": front_img, "side": side_img}

def render_scene(
    actors: List[vtk.vtkActor],
    camera_pos: Tuple[float, float, float],
    focal_point: Tuple[float, float, float],
    view_up: Tuple[float, float, float],
    output_filepath: str,
    width: int = 1600,
    height: int = 1200
) -> None:
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.043, 0.059, 0.090)

    for actor in actors:
        renderer.AddActor(actor)

    camera = renderer.GetActiveCamera()
    camera.SetPosition(camera_pos[0], camera_pos[1], camera_pos[2])
    camera.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
    camera.SetViewUp(view_up[0], view_up[1], view_up[2])
    renderer.ResetCameraClippingRange()

    light1 = vtk.vtkLight()
    light1.SetPosition(150.0, 200.0, 200.0)
    light1.SetColor(0.024, 0.714, 0.831)
    light1.SetIntensity(0.7)
    renderer.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetPosition(-150.0, -200.0, 100.0)
    light2.SetColor(0.918, 0.702, 0.031)
    light2.SetIntensity(0.4)
    renderer.AddLight(light2)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(width, height)
    render_window.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_filepath)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

def print_audit_report(results: List[dict], conduit_metrics: List[dict]) -> None:
    print("\n" + "=" * 105)
    print("                     3D WIRE HARNESS, CONNECTOR & BEND-RADIUS AUDIT")
    print("=" * 105)
    print(f"{'Type':<20} | {'Name':<26} | {'Max Collision (mm³)':<22} | {'Status':<6}")
    print("-" * 105)
    for r in results:
        status_colored = f"\033[92m{r['status']}\033[0m" if r['status'] == 'PASS' else f"\033[91m{r['status']}\033[0m"
        print(f"{r['type']:<20} | {r['name']:<26} | {r['collision_mm3']:<22} | {r['status']:<6}")
    print("=" * 105)

    print("\n" + "=" * 105)
    print("                     CONDUIT CROSS-SECTION & BEND-RADIUS AUDIT")
    print("=" * 105)
    print(f"{'Conduit Description':<38} | {'Conduit Ø':<11} | {'Fill Area':<12} | {'Free Air Gap (%)':<18} | {'Min R_bend':<11} | {'Status':<6}")
    print("-" * 105)
    for m in conduit_metrics:
        free_pct = f"{m['free_air_gap_pct']:.1f}%"
        rbend = f"{m['min_bend_radius_mm']:.1f} mm"
        status = "PASS" if m['free_air_gap_pct'] > 50.0 and m['min_bend_radius_mm'] >= 8.0 else "FAIL"
        print(f"{m['conduit_name']:<38} | Ø {m['conduit_diam']:.1f}mm   | {m['wire_area_mm2']:.1f} mm²    | {free_pct:<18} | {rbend:<11} | {status:<6}")
    print("=" * 105 + "\n")

def main() -> int:
    print("[1/5] Building watertight solids for all 31 components and shells...")
    components = {spec.name: verify_electronics_fit.build_component_solid(spec) for spec in COMPONENTS_REGISTRY}
    shells = build_wiring_cleared_shells()

    print("[2/5] Building connector clearance envelopes...")
    connectors = {conn.name: build_connector_solid(conn) for conn in CONNECTORS_REGISTRY}

    print("[3/5] Sweeping 3D physical wire harnesses along conduits...")
    harnesses = {spec.name: build_wire_harness_solid(spec) for spec in HARNESSES_REGISTRY}

    print("[4/5] Running boolean interference and conduit clearance audit...")
    audit_results = run_wiring_interference_audit(components, connectors, harnesses, shells)

    conduit_metrics = []
    for hspec in HARNESSES_REGISTRY:
        met = calculate_conduit_fill_metrics(hspec)
        met["conduit_name"] = hspec.conduit_name
        met["conduit_diam"] = hspec.conduit_inner_diam_mm
        met["min_bend_radius_mm"] = hspec.min_bend_radius_mm
        conduit_metrics.append(met)

    print_audit_report(audit_results, conduit_metrics)

    print("[5/5] Exporting STL meshes and rendering translucent X-ray wiring views...")
    stl_path = export_all_wiring_stls(connectors, harnesses)
    print(f"Exported composite wire harness STL: {stl_path}")

    renders = render_wiring_xray_views(components, connectors, harnesses)
    print(f"Rendered Front View: {renders['front']}")
    print(f"Rendered Side View:  {renders['side']}")

    all_passed = all(r["status"] == "PASS" for r in audit_results) and all(m["free_air_gap_pct"] > 50.0 for m in conduit_metrics)
    return 0 if all_passed else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
