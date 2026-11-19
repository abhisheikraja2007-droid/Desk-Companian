from __future__ import annotations
import os
from typing import Dict, Tuple, List
import vtk

def create_part_actor(
    stl_path: str,
    color: Tuple[float, float, float],
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    opacity: float = 1.0
) -> vtk.vtkActor:
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
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
    actor.GetProperty().SetSpecular(0.4)
    actor.GetProperty().SetSpecularPower(35.0)
    actor.GetProperty().SetAmbient(0.25)
    actor.GetProperty().SetDiffuse(0.75)
    actor.GetProperty().SetOpacity(opacity)
    actor.SetPosition(position[0], position[1], position[2])

    if rotation != (0.0, 0.0, 0.0):
        actor.RotateZ(rotation[2])
        actor.RotateY(rotation[1])
        actor.RotateX(rotation[0])

    return actor

def render_scene(
    actors: List[vtk.vtkActor],
    camera_pos: Tuple[float, float, float],
    focal_point: Tuple[float, float, float],
    view_up: Tuple[float, float, float],
    output_path: str,
    resolution: Tuple[int, int] = (1600, 1200)
) -> None:
    renderer = vtk.vtkRenderer()
    for actor in actors:
        renderer.AddActor(actor)

    renderer.SetBackground(0.043, 0.059, 0.090) # #0B0F17
    renderer.SetBackground2(0.067, 0.094, 0.153) # #111827 gradient
    renderer.SetGradientBackground(True)

    camera = renderer.GetActiveCamera()
    camera.SetPosition(camera_pos[0], camera_pos[1], camera_pos[2])
    camera.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
    camera.SetViewUp(view_up[0], view_up[1], view_up[2])

    light1 = vtk.vtkLight()
    light1.SetPosition(camera_pos[0] + 60, camera_pos[1] + 160, camera_pos[2] + 100)
    light1.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
    light1.SetColor(0.024, 0.714, 0.831) # Cyan accent light
    light1.SetIntensity(0.85)
    renderer.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetPosition(camera_pos[0] - 160, camera_pos[1] - 80, camera_pos[2] + 50)
    light2.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
    light2.SetColor(0.918, 0.702, 0.031) # Yellow fill light
    light2.SetIntensity(0.45)
    renderer.AddLight(light2)

    light3 = vtk.vtkLight()
    light3.SetPosition(0.0, 0.0, 320.0)
    light3.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
    light3.SetColor(1.0, 1.0, 1.0)
    light3.SetIntensity(0.6)
    renderer.AddLight(light3)

    ren_win = vtk.vtkRenderWindow()
    ren_win.SetOffScreenRendering(1)
    ren_win.AddRenderer(renderer)
    ren_win.SetSize(resolution[0], resolution[1])
    ren_win.SetMultiSamples(8)

    ren_win.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(ren_win)
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_path)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

def build_assembly_actors(is_exploded: bool = False) -> List[vtk.vtkActor]:
    factor = 1.0 if is_exploded else 0.0
    stl_dir = os.path.join("exports", "stl")

    configs = [
        ("base_link.stl", (0.118, 0.161, 0.231), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("left_wheel.stl", (0.059, 0.090, 0.165), (-49.0 - factor * 25.0, 0.0, 17.0), (0.0, 0.0, 0.0)),
        ("right_wheel.stl", (0.059, 0.090, 0.165), (49.0 + factor * 25.0, 0.0, 17.0), (0.0, 0.0, 0.0)),
        ("torso_shell.stl", (0.078, 0.722, 0.651), (0.0, 0.0, 34.0 + factor * 25.0), (0.0, 0.0, 0.0)),
        ("left_arm.stl", (0.024, 0.714, 0.831), (-48.5 - factor * 20.0, 2.0, 78.0 + factor * 25.0), (0.0, 10.0, 0.0)),
        ("left_pincer.stl", (0.918, 0.702, 0.031), (-53.01 - factor * 20.0, 2.0, 52.4 + factor * 25.0), (0.0, 10.0, 0.0)),
        ("right_arm.stl", (0.024, 0.714, 0.831), (48.5 + factor * 20.0, 2.0, 78.0 + factor * 25.0), (0.0, -10.0, 0.0)),
        ("right_pincer.stl", (0.918, 0.702, 0.031), (53.01 + factor * 20.0, 2.0, 52.4 + factor * 25.0), (0.0, -10.0, 0.0)),
        ("neck_yaw_turret.stl", (0.024, 0.714, 0.831), (0.0, 0.0, 96.0 + factor * 50.0), (0.0, 0.0, 0.0)),
        ("head_assembly.stl", (0.051, 0.580, 0.533), (0.0, 0.0, 108.0 + factor * 75.0), (0.0, 0.0, 0.0)),
        ("left_ear.stl", (0.918, 0.702, 0.031), (-32.0 - factor * 15.0, 2.0, 138.0 + factor * 95.0), (49.08, -20.0, 0.0)),
        ("right_ear.stl", (0.918, 0.702, 0.031), (32.0 + factor * 15.0, 2.0, 138.0 + factor * 95.0), (49.08, 20.0, 0.0)),
        ("left_eye.stl", (0.05, 0.95, 0.95), (-13.5, 35.0, 125.0 + factor * 75.0), (0.0, 0.0, 0.0)),
        ("right_eye.stl", (0.05, 0.95, 0.95), (13.5, 35.0, 125.0 + factor * 75.0), (0.0, 0.0, 0.0)),
    ]

    actors = []
    for filename, color, pos, rot in configs:
        path = os.path.join(stl_dir, filename)
        if os.path.exists(path):
            actors.append(create_part_actor(path, color, pos, rot))
    return actors

def main() -> None:
    out_dir = os.path.join("exports", "renders")
    os.makedirs(out_dir, exist_ok=True)

    iso_actors = build_assembly_actors(is_exploded=False)
    iso_path = os.path.join(out_dir, "desk_pet_isometric.png")
    render_scene(
        iso_actors,
        camera_pos=(85.0, 220.0, 125.0),
        focal_point=(0.0, 0.0, 82.0),
        view_up=(0.0, 0.0, 1.0),
        output_path=iso_path
    )
    print(f"Rendered: {iso_path}")

    front_actors = build_assembly_actors(is_exploded=False)
    front_path = os.path.join(out_dir, "desk_pet_front.png")
    render_scene(
        front_actors,
        camera_pos=(0.0, 280.0, 90.0),
        focal_point=(0.0, 0.0, 85.0),
        view_up=(0.0, 0.0, 1.0),
        output_path=front_path
    )
    print(f"Rendered: {front_path}")

    side_actors = build_assembly_actors(is_exploded=False)
    side_path = os.path.join(out_dir, "desk_pet_side.png")
    render_scene(
        side_actors,
        camera_pos=(280.0, 0.0, 90.0),
        focal_point=(0.0, 0.0, 85.0),
        view_up=(0.0, 0.0, 1.0),
        output_path=side_path
    )
    print(f"Rendered: {side_path}")

    exp_actors = build_assembly_actors(is_exploded=True)
    exp_path = os.path.join(out_dir, "desk_pet_exploded.png")
    render_scene(
        exp_actors,
        camera_pos=(160.0, 260.0, 160.0),
        focal_point=(0.0, 0.0, 95.0),
        view_up=(0.0, 0.0, 1.0),
        output_path=exp_path
    )
    print(f"Rendered: {exp_path}")

if __name__ == "__main__":
    main()
