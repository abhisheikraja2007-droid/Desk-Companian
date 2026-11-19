# Locked 3D Print-Ready STL Pack: Desk Companion Robot (8-DOF)

This directory contains the final, manufacturing-locked STL meshes oriented specifically for direct FDM 3D printing with minimal support structures and optimal layer adhesion.

## Recommended FDM Slicer Settings (Bambu Studio / PrusaSlicer / OrcaSlicer)
- **Material**: PETG or PLA+ / Tough PLA (Arms and Pincers: PETG or ABS for impact durability; Pincer Grip Pads: 85A TPU)
- **Layer Height**: 0.16 mm (Optimal compromise for curved spline surfaces and thread tolerances)
- **Perimeters / Wall Loops**: 4 walls (≥ 1.6 mm solid perimeter shell)
- **Top / Bottom Solid Layers**: 5 top, 4 bottom
- **Infill**: 20% Gyroid or Adaptive Cubic (Uniform isotropic shear resistance)
- **Support Type**: Tree (Organic / Slim) Supports with 45° threshold angle
- **Bed Temperature**: 60°C (PLA) / 75°C (PETG)
- **Nozzle Temperature**: 215°C (PLA) / 240°C (PETG)

## Print-Ready Parts Summary
| Part Name | Build Orientation | Bed Extents (mm) | Part Vol (cm³) | Support Vol (cm³) | Rating |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `torso_front_print_ready.stl` | Split-face down on bed (Y=0 -> Bed Z=0) | 97.0×96.0×43.8 | 26.89 | 45.09 | ACCEPTABLE (Organic Supports Recommended) |
| `torso_rear_print_ready.stl` | Split-face down on bed (Y=0 -> Bed Z=0) | 97.0×105.0×46.0 | 35.03 | 47.51 | ACCEPTABLE (Organic Supports Recommended) |
| `head_front_print_ready.stl` | Split-face down on bed (Y=4 -> Bed Z=0) | 71.6×40.8×33.2 | 12.36 | 8.33 | ACCEPTABLE (Organic Supports Recommended) |
| `head_rear_print_ready.stl` | Split-face down on bed (Y=4 -> Bed Z=0) | 71.6×40.4×36.0 | 1.0 | 4.6 | ACCEPTABLE (Organic Supports Recommended) |
| `base_link_print_ready.stl` | Bottom deck down on bed (Z=0 -> Bed Z=0) | 109.0×115.0×34.0 | 1.0 | 337.9 | ACCEPTABLE (Organic Supports Recommended) |
| `left_wheel_print_ready.stl` | Outer hub face flat on bed | 34.0×34.0×12.0 | 1.0 | 0.0 | EXCELLENT (Minimal / Bridging Only) |
| `right_wheel_print_ready.stl` | Outer hub face flat on bed | 34.0×34.0×12.0 | 1.0 | 0.0 | EXCELLENT (Minimal / Bridging Only) |
| `neck_yaw_turret_print_ready.stl` | Bottom bearing boss flat on bed | 26.0×26.0×16.0 | 4.94 | 1.65 | GOOD (Standard Tree Supports) |
| `left_arm_print_ready.stl` | Lateral flat flank on bed | 33.8×9.6×8.0 | 1.8 | 0.11 | EXCELLENT (Minimal / Bridging Only) |
| `right_arm_print_ready.stl` | Lateral flat flank on bed | 33.8×9.6×8.0 | 1.8 | 0.11 | EXCELLENT (Minimal / Bridging Only) |
| `left_pincer_print_ready.stl` | Lateral hinge flat face on bed | 20.0×11.0×4.5 | 0.37 | 0.04 | EXCELLENT (Minimal / Bridging Only) |
| `right_pincer_print_ready.stl` | Lateral hinge flat face on bed | 20.0×11.0×4.5 | 0.37 | 0.04 | EXCELLENT (Minimal / Bridging Only) |
| `left_ear_print_ready.stl` | Cylindrical hinge pivot flat on bed | 45.5×15.0×3.5 | 0.79 | 0.68 | ACCEPTABLE (Organic Supports Recommended) |
| `right_ear_print_ready.stl` | Cylindrical hinge pivot flat on bed | 45.5×15.0×3.5 | 0.79 | 0.68 | ACCEPTABLE (Organic Supports Recommended) |
| `left_eye_print_ready.stl` | Flat back mounting surface on bed | 11.0×1.8×9.0 | 0.13 | 0.02 | EXCELLENT (Minimal / Bridging Only) |
| `right_eye_print_ready.stl` | Flat back mounting surface on bed | 11.0×1.8×9.0 | 0.13 | 0.02 | EXCELLENT (Minimal / Bridging Only) |
