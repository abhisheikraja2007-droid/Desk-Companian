# Desk-Companian: 8-DOF Desk Companion Robot

Parametric CAD geometry, physical wire harness routing, electronics fit audit, Wokwi ESP32-S3 firmware simulation, simulation URDF, and Design-for-Manufacturing (DFM) verification suite for an 8-DOF companion robot.

## Features
- **8 Active Degrees of Freedom**: 2 differential drive wheels (N20 motors), 2-DOF neck (yaw/pitch SCS0009 serial servos), 2-DOF arms (with 3.7g micro pincers), and 2-DOF ears.
- **Serviceable DFA Clamshells**: Tier 2 torso and Tier 3 head split along planar seams with alignment lips, cable passthrough conduits, and hidden rear fasteners.
- **3D Wire Harnesses & Routing**: 5 physical swept wire bundles + 24-pin FPC camera ribbon with connector keep-out envelopes and verified minimum bend radii ($R_{\min} \ge 9.5\text{ mm}$).
- **Wokwi ESP32-S3 Circuit Simulation**: Full hardware emulation in `./wokwi_sim/` with dual SSD1306 OLED eyes, MPU6050 IMU, TTP223 crown touch sensor, 4× TCRT5000 cliff interrupts (<2 ms latency), and TB6612FNG motor drive.
- **Real-Time 3D Telemetry Bridge**: Serial-to-WebSocket bridge connecting Wokwi live simulation (port 4000) to the Three.js 3D viewer (ws://localhost:8765) with 20 Hz state synchronization.
- **Simulation Assets**: URDF with physical inertia tensors for NVIDIA Isaac Lab and MuJoCo.
- **Interactive 3D Viewer**: WebGL HUD with joint sliders, X-Ray mode, 3D wiring toggles, and clamshell disassembly view.

## Verification & Simulation Tools
- `python build_desk_pet.py`: Builds CAD assembly, exports STEP, STL, and URDF models.
- `python verify_electronics_fit.py`: Audits internal component clearances (31 parts) and screwdriver tool access paths.
- `python verify_wiring_and_fit.py`: Sweeps 3D wire harnesses, validates connector keep-outs, asserts 0.00 mm³ collision, and audits conduit free-air gaps.
- `python verify_wiring_and_record_proof.py`: Automated Playwright headless Chromium runner recording 15-second visual proof of clamshells and wire harnesses.
- `python dfm_check.py`: Runs wall thickness checks, aperture verification, and flat-bed orientation analysis.
- `python generate_standalone_viewer.py`: Generates the self-contained standalone `exports/viewer.html`.
- `python wokwi_sim/launch_and_record_3d.py`: Unified launcher orchestrating the HTTP server, WebSocket bridge, and kinematics playback.

## Repository File Structure
- `build_desk_pet.py`: Parametric CAD model and URDF exporter.
- `verify_electronics_fit.py`: DFA clearance checker for all 31 internal electronic modules.
- `verify_wiring_and_fit.py`: 3D solid wire harness sweep and conduit clearance auditor.
- `verify_wiring_and_record_proof.py`: Playwright video proof recorder and transcoder.
- `dfm_check.py`: DFM analysis and print-ready STL exporter.
- `generate_standalone_viewer.py`: Self-contained Three.js 3D HTML viewer builder.
- `render_robot_views.py`: Image render generator.
- `wokwi_sim/`: Complete Wokwi ESP32-S3 circuit simulation suite:
  - `diagram.json`: Electronic schematic and breadboard layout wiring.
  - `platformio.ini`: Toolchain dependencies and build configuration.
  - `wokwi.toml`: RFC2217 port 4000 serial mapping and firmware artifacts.
  - `src/main.cpp`: ESP32-S3 Arduino C++ firmware (emotions, cliff ISR, 20 Hz JSON telemetry).
  - `live_bridge.py`: RFC2217 serial-to-WebSocket telemetry server.
  - `launch_and_record_3d.py`: Unified local server and video demonstration recorder.
  - `run_full_system_test.py`: Headless automated verification suite.
- `exports/`:
  - `viewer.html`: Interactive 3D viewer with live telemetry and 3D wiring harnesses.
  - `print_ready/`: Pre-oriented STL files for 3D printing.
  - `stl/`: Outer assembly STL meshes and `internals/` subfolder (31 component solids + 6 harness solids).
  - `step/`: Complete CAD assembly STEP models.
  - `urdf/`: Robot description format with kinematics and visual/collision geometries.
  - `renders/`: High-resolution orthogonal, isometric, exploded, and X-ray renders.
  - `videos/`: Recorded MP4/WebM video proofs (`wiring_fit_proof.mp4`, `desk_pet_live_demo.mp4`).
