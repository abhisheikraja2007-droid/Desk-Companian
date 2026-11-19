# Desk-Companian: 8-DOF Desk Companion Robot

Parametric CAD geometry, simulation URDF, and Design-for-Manufacturing (DFM) verification suite for an 8-DOF companion robot.

## Features
- **8 Active Degrees of Freedom**: 2 differential drive wheels, 2-DOF neck (yaw/pitch), 2-DOF arms, and 2-DOF ears.
- **Serviceable Clamshells**: Tier 2 torso and Tier 3 head split along planar seams with alignment lips and hidden rear fasteners.
- **Physical Cutouts**: Integrated USB-C port, front speaker grille, microphone pinhole, slide switch, and reset hole.
- **Simulation Assets**: URDF with physical inertia tensors for NVIDIA Isaac Lab and MuJoCo.
- **Interactive 3D Viewer**: WebGL HUD with joint sliders, X-Ray mode, and clamshell disassembly view.

## Verification
- `python build_desk_pet.py`: Builds CAD assembly, exports STEP, STL, and URDF models.
- `python verify_electronics_fit.py`: Audits internal component clearances and screwdriver access.
- `python dfm_check.py`: Runs wall thickness checks, aperture verification, and flat-bed orientation analysis.
- `python generate_standalone_viewer.py`: Generates the standalone `exports/viewer.html`.

## File Structure
- `build_desk_pet.py`: Parametric CAD model and URDF exporter.
- `verify_electronics_fit.py`: DFA clearance checker.
- `dfm_check.py`: DFM analysis and print-ready STL exporter.
- `generate_standalone_viewer.py`: 3D HTML viewer builder.
- `render_robot_views.py`: Image render generator.
- `exports/`:
  - `viewer.html`: Interactive 3D viewer.
  - `print_ready/`: Pre-oriented STL files for 3D printing.
  - `stl/`: STL meshes.
  - `step/`: STEP files.
  - `urdf/`: Robot description format.
  - `renders/`: High-resolution renders.
