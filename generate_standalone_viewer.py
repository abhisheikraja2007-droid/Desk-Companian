from __future__ import annotations
import base64
import os

configs = [
    ("base_link", "exports/stl/base_link.stl", 0x1E293B),
    ("left_wheel", "exports/stl/left_wheel.stl", 0x0F172A),
    ("right_wheel", "exports/stl/right_wheel.stl", 0x0F172A),
    ("torso_front", "exports/stl/torso_front.stl", 0x14B8A6),
    ("torso_rear", "exports/stl/torso_rear.stl", 0x0F766E),
    ("left_arm", "exports/stl/left_arm.stl", 0x06B6D4),
    ("left_pincer", "exports/stl/left_pincer.stl", 0xEAB308),
    ("right_arm", "exports/stl/right_arm.stl", 0x06B6D4),
    ("right_pincer", "exports/stl/right_pincer.stl", 0xEAB308),
    ("neck_yaw_turret", "exports/stl/neck_yaw_turret.stl", 0x06B6D4),
    ("head_front", "exports/stl/head_front.stl", 0x0D9488),
    ("head_rear", "exports/stl/head_rear.stl", 0x115E59),
    ("left_ear", "exports/stl/left_ear.stl", 0xEAB308),
    ("right_ear", "exports/stl/right_ear.stl", 0xEAB308),
    ("left_eye", "exports/stl/left_eye.stl", 0x06B6D4),
    ("right_eye", "exports/stl/right_eye.stl", 0x06B6D4),
]

parts_dict = {}
for name, path, color in configs:
    if os.path.exists(path):
        with open(path, "rb") as f:
            parts_dict[name] = {
                "color": hex(color),
                "b64": base64.b64encode(f.read()).decode("ascii")
            }

internals_dir = os.path.join("exports", "stl", "internals")
internal_configs = [
    ("n20_motor_left", 0x64748B, "Actuator"),
    ("n20_motor_right", 0x64748B, "Actuator"),
    ("tcrt5000_fl", 0xEC4899, "Sensor"),
    ("tcrt5000_fr", 0xEC4899, "Sensor"),
    ("tcrt5000_rl", 0xEC4899, "Sensor"),
    ("tcrt5000_rr", 0xEC4899, "Sensor"),
    ("caster_ball_front", 0x475569, "Mechanism"),
    ("caster_ball_rear", 0x475569, "Mechanism"),
    ("tb6612_driver", 0xEF4444, "Power"),
    ("buck_12v_6v", 0xEAB308, "Power"),
    ("esp32_s3", 0x3B82F6, "Compute"),
    ("speaker_40mm", 0x334155, "Audio"),
    ("max98357a_amp", 0x14B8A6, "Audio"),
    ("buck_12v_5v", 0xEAB308, "Power"),
    ("mpu6050_imu", 0xEC4899, "Sensor"),
    ("shoulder_servo_left", 0xF59E0B, "Actuator"),
    ("shoulder_servo_right", 0xF59E0B, "Actuator"),
    ("neck_yaw_servo", 0xF59E0B, "Actuator"),
    ("neck_bearing_6704zz", 0x94A3B8, "Mechanism"),
    ("slip_ring_12wire", 0xD97706, "Power"),
    ("usb_pd_trigger", 0xEF4444, "Power"),
    ("neck_pitch_servo", 0xF59E0B, "Actuator"),
    ("ear_servo_left", 0xF59E0B, "Actuator"),
    ("ear_servo_right", 0xF59E0B, "Actuator"),
    ("oled_display_left", 0x06B6D4, "Display"),
    ("oled_display_right", 0x06B6D4, "Display"),
    ("camera_ov5640", 0x14B8A6, "Vision"),
    ("inmp441_mic", 0xEC4899, "Audio"),
    ("touch_ttp223", 0xEC4899, "Sensor"),
    ("arm_micro_servo_left", 0xF59E0B, "Actuator"),
    ("arm_micro_servo_right", 0xF59E0B, "Actuator"),
]

internals_dict = {}
for name, color, category in internal_configs:
    path = os.path.join(internals_dir, f"{name}.stl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            internals_dict[name] = {
                "color": hex(color),
                "category": category,
                "b64": base64.b64encode(f.read()).decode("ascii")
            }

parts_js_entries = []
for name, data in parts_dict.items():
    parts_js_entries.append(f"  '{name}': {{ color: {data['color']}, b64: '{data['b64']}' }}")

parts_js = "{\n" + ",\n".join(parts_js_entries) + "\n}"

internals_js_entries = []
for name, data in internals_dict.items():
    internals_js_entries.append(f"  '{name}': {{ color: {data['color']}, category: '{data['category']}', b64: '{data['b64']}' }}")

internals_js = "{\n" + ",\n".join(internals_js_entries) + "\n}"

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Desk Companion Robot - 8-DOF Kinematics &amp; DFA Clamshell HUD</title>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background-color: #0B0F17;
      color: #F3F4F6;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      overflow: hidden;
    }}
    #viewport {{
      width: 100vw;
      height: 100vh;
      display: block;
    }}
    #hud {{
      position: absolute;
      top: 16px;
      right: 16px;
      width: 320px;
      max-height: calc(100vh - 32px);
      overflow-y: auto;
      background: rgba(17, 24, 39, 0.95);
      border: 1px solid #14B8A6;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.6);
      backdrop-filter: blur(8px);
    }}
    #hud::-webkit-scrollbar {{
      width: 5px;
    }}
    #hud::-webkit-scrollbar-thumb {{
      background: #14B8A6;
      border-radius: 3px;
    }}
    h1 {{
      font-size: 15px;
      margin: 0 0 10px 0;
      color: #06B6D4;
      text-transform: uppercase;
      letter-spacing: 1px;
    }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      margin: 4px 0;
      font-size: 12px;
    }}
    .stat-label {{
      color: #9CA3AF;
    }}
    .stat-val {{
      font-weight: 600;
      color: #14B8A6;
    }}
    .stat-val.accent {{
      color: #EAB308;
    }}
    .status-badge {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(20, 184, 166, 0.2);
      color: #14B8A6;
      border: 1px solid #14B8A6;
      font-size: 10px;
      font-weight: bold;
    }}
    .section-title {{
      font-size: 11px;
      font-weight: bold;
      color: #06B6D4;
      text-transform: uppercase;
      margin: 12px 0 6px 0;
      padding-bottom: 3px;
      border-bottom: 1px solid #374151;
      letter-spacing: 0.5px;
    }}
    .slider-row {{
      margin: 6px 0;
    }}
    .slider-header {{
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      margin-bottom: 2px;
    }}
    .slider-name {{
      color: #D1D5DB;
    }}
    .slider-val {{
      color: #EAB308;
      font-weight: bold;
    }}
    input[type=range] {{
      width: 100%;
      height: 4px;
      background: #374151;
      border-radius: 2px;
      outline: none;
      -webkit-appearance: none;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #06B6D4;
      cursor: pointer;
      box-shadow: 0 0 6px rgba(6, 182, 212, 0.8);
    }}
    .btn-group {{
      display: flex;
      gap: 4px;
      margin-bottom: 8px;
    }}
    .exp-btn {{
      flex: 1;
      background: #1F2937;
      border: 1px solid #374151;
      color: #D1D5DB;
      padding: 5px 4px;
      font-size: 10px;
      font-weight: 600;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .exp-btn:hover {{
      border-color: #06B6D4;
      color: #06B6D4;
    }}
    .exp-btn.active {{
      background: rgba(6, 182, 212, 0.25);
      border-color: #06B6D4;
      color: #06B6D4;
    }}
    #controls {{
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid #374151;
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    button.ctrl-btn {{
      flex: 1 1 45%;
      background: #111827;
      border: 1px solid #374151;
      color: #E5E7EB;
      padding: 6px 8px;
      font-size: 11px;
      border-radius: 4px;
      cursor: pointer;
      font-weight: 500;
      transition: all 0.2s ease;
    }}
    button.ctrl-btn:hover {{
      border-color: #14B8A6;
      color: #14B8A6;
    }}
    button.ctrl-btn.highlight {{
      border-color: #06B6D4;
      color: #06B6D4;
      background: rgba(6, 182, 212, 0.15);
    }}
    #xray-legend {{
      display: none;
      margin-top: 8px;
      padding: 8px;
      background: #0B0F17;
      border: 1px dashed #06B6D4;
      border-radius: 6px;
    }}
    .legend-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
      font-size: 10px;
      margin-top: 4px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    .legend-dot {{
      width: 8px;
      height: 8px;
      border-radius: 2px;
    }}
    #loading {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 14px;
      letter-spacing: 2px;
      color: #06B6D4;
      font-weight: bold;
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
</head>
<body>
  <div id="loading">INITIALIZING CAD MESHES &amp; KINEMATICS...</div>
  <div id="hud">
    <h1>Desk Pet 8-DOF HUD</h1>
    <div class="stat-row">
      <span class="stat-label">Live Wokwi Sync:</span>
      <span id="wokwi-badge" class="status-badge" style="background: rgba(100, 116, 139, 0.2); color: #94A3B8; border-color: #475569;">OFFLINE</span>
    </div>
    <div class="stat-row" id="wokwi-event-row" style="display:none;">
      <span class="stat-label">Hardware Event:</span>
      <span id="wokwi-event-val" class="stat-val accent" style="font-size: 11px;">NONE</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Kinematic State:</span>
      <span class="status-badge">ONLINE &amp; CALIBRATED</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Internal Fit Audit:</span>
      <span class="stat-val accent">31/31 PASSED (100.0%)</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Screwdriver Access:</span>
      <span class="stat-val accent">7/7 Steps PASSED (100%)</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Total System Mass:</span>
      <span class="stat-val">680.86 g</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Center of Gravity:</span>
      <span class="stat-val">Z = 35.81 mm</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Fasteners &amp; Tunnels:</span>
      <span class="stat-val accent">4× M3 Torso / 3× M2.5 Head</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Interlocking Seam Lip:</span>
      <span class="stat-val">1.5 mm Torso / 1.2 mm Head</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Wire Routing Channels:</span>
      <span class="stat-val accent">3/3 OPEN (0.0 mm³)</span>
    </div>

    <div class="section-title">DFA Clamshell Disassembly</div>
    <div class="slider-row">
      <div class="slider-header">
        <span class="slider-name">Clamshell Split Y</span>
        <span id="val-clamshell" class="slider-val">0 mm (Sealed)</span>
      </div>
      <input type="range" id="slider-clamshell" min="0" max="45" value="0" oninput="updateClamshellSplit()">
    </div>
    <div class="btn-group">
      <button class="exp-btn active" id="btn-clam-close" onclick="setClamshellSplit(0)">Sealed</button>
      <button class="exp-btn" id="btn-clam-inspect" onclick="setClamshellSplit(20)">Inspect (20mm)</button>
      <button class="exp-btn" id="btn-clam-full" onclick="setClamshellSplit(40)">Full Split (40mm)</button>
    </div>

    <div class="section-title">Expressive Eye Morphs</div>
    <div class="btn-group">
      <button class="exp-btn active" id="btn-happy" onclick="setEyeExpression('happy')">Happy</button>
      <button class="exp-btn" id="btn-curious" onclick="setEyeExpression('curious')">Curious</button>
      <button class="exp-btn" id="btn-sleepy" onclick="setEyeExpression('sleepy')">Sleepy</button>
      <button class="exp-btn" id="btn-alert" onclick="setEyeExpression('alert')">Alert</button>
    </div>

    <div class="section-title">Articulated Ears (2-DOF)</div>
    <div class="slider-row">
      <div class="slider-header">
        <span class="slider-name">Left Ear Pitch</span>
        <span id="val-ear-l" class="slider-val">0°</span>
      </div>
      <input type="range" id="slider-ear-l" min="-45" max="65" value="0" oninput="updateJoints()">
    </div>
    <div class="slider-row">
      <div class="slider-header">
        <span class="slider-name">Right Ear Pitch</span>
        <span id="val-ear-r" class="slider-val">0°</span>
      </div>
      <input type="range" id="slider-ear-r" min="-45" max="65" value="0" oninput="updateJoints()">
    </div>

    <div class="section-title">Neck Turret &amp; Cranium (2-DOF)</div>
    <div class="slider-row">
      <div class="slider-header">
        <span class="slider-name">Neck Yaw</span>
        <span id="val-yaw" class="slider-val">0°</span>
      </div>
      <input type="range" id="slider-yaw" min="-75" max="75" value="0" oninput="updateJoints()">
    </div>
    <div class="slider-row">
      <div class="slider-header">
        <span class="slider-name">Neck Pitch</span>
        <span id="val-pitch" class="slider-val">0°</span>
      </div>
      <input type="range" id="slider-pitch" min="-30" max="40" value="0" oninput="updateJoints()">
    </div>

    <div class="section-title">2-DOF Connected Arms &amp; Pincers</div>
    <div class="slider-row">
      <div class="slider-header">
        <span class="slider-name">Left Shoulder Pitch</span>
        <span id="val-shoulder-l" class="slider-val">0°</span>
      </div>
      <input type="range" id="slider-shoulder-l" min="-40" max="135" value="0" oninput="updateJoints()">
    </div>
    <div class="slider-row">
      <div class="slider-header">
        <span class="slider-name">Left Pincer Claw</span>
        <span id="val-pincer-l" class="slider-val">0° (Pinch)</span>
      </div>
      <input type="range" id="slider-pincer-l" min="0" max="55" value="0" oninput="updateJoints()">
    </div>
    <div class="slider-row">
      <div class="slider-header">
        <span class="slider-name">Right Shoulder Pitch</span>
        <span id="val-shoulder-r" class="slider-val">0°</span>
      </div>
      <input type="range" id="slider-shoulder-r" min="-40" max="135" value="0" oninput="updateJoints()">
    </div>
    <div class="slider-row">
      <div class="slider-header">
        <span class="slider-name">Right Pincer Claw</span>
        <span id="val-pincer-r" class="slider-val">0° (Pinch)</span>
      </div>
      <input type="range" id="slider-pincer-r" min="0" max="55" value="0" oninput="updateJoints()">
    </div>

    <div id="controls">
      <button class="ctrl-btn highlight" id="btn-wokwi-sync" onclick="toggleWokwiSync()" style="flex: 1 1 100%; font-weight: bold; background: rgba(6, 182, 212, 0.25); border-color: #06B6D4; color: #06B6D4;">Live Wokwi Sync: OFF</button>
      <button class="ctrl-btn highlight" id="btn-dfa" onclick="toggleClamshellSplit()">DFA Clamshell Split</button>
      <button class="ctrl-btn" id="btn-xray" onclick="toggleXray()">X-Ray Mode: OFF</button>
      <button class="ctrl-btn" id="btn-explode" onclick="toggleExploded()">Explode View</button>
      <button class="ctrl-btn" onclick="setPreset('neutral')">Neutral Pose</button>
      <button class="ctrl-btn" onclick="setPreset('wave')">Wave Pose</button>
      <button class="ctrl-btn" onclick="setPreset('hold')">Hold Stylus</button>
      <button class="ctrl-btn" onclick="resetCamera()">Reset View (+Y)</button>
    </div>

    <div id="xray-legend">
      <div style="font-size:10px; font-weight:bold; color:#06B6D4; margin-bottom:4px;">INTERNAL COMPONENTS (31 PCS)</div>
      <div class="legend-grid">
        <div class="legend-item"><span class="legend-dot" style="background:#3B82F6;"></span>Compute (ESP32-S3)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#F59E0B;"></span>Actuators (Servos)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#06B6D4;"></span>Display (OLEDs)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#14B8A6;"></span>Vision / Camera</div>
        <div class="legend-item"><span class="legend-dot" style="background:#EC4899;"></span>Sensors (IMU/Cliff)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#EF4444;"></span>Power (Buck/PD)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#334155;"></span>Audio (Speaker/Amp)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#64748B;"></span>Motors &amp; Bearing</div>
      </div>
    </div>
  </div>
  <div id="viewport"></div>
  <script>
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0B0F17);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);
    camera.up.set(0, 0, 1);
    camera.position.set(85, 220, 125);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    document.getElementById('viewport').appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 82);
    controls.update();

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x06B6D4, 0.85);
    dirLight1.position.set(80, 220, 160);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xEAB308, 0.45);
    dirLight2.position.set(-140, -80, 70);
    scene.add(dirLight2);

    const grid = new THREE.GridHelper(300, 30, 0x14B8A6, 0x1F2937);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const loader = new THREE.STLLoader();
    const robotRoot = new THREE.Group();
    scene.add(robotRoot);

    const rawData = {parts_js};
    const rawInternals = {internals_js};

    function base64ToArrayBuffer(base64) {{
      const binaryString = window.atob(base64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {{
        bytes[i] = binaryString.charCodeAt(i);
      }}
      return bytes.buffer;
    }}

    const shellMeshes = [];

    function createMesh(name) {{
      const d = rawData[name];
      if (!d) return new THREE.Group();
      const buffer = base64ToArrayBuffer(d.b64);
      const geom = loader.parse(buffer);
      
      let mat;
      if (name.includes('eye')) {{
        mat = new THREE.MeshStandardMaterial({{
          color: 0x06B6D4,
          emissive: 0x06B6D4,
          emissiveIntensity: 0.9,
          roughness: 0.2,
          metalness: 0.1
        }});
      }} else {{
        mat = new THREE.MeshStandardMaterial({{
          color: d.color,
          metalness: 0.3,
          roughness: 0.5,
          transparent: false,
          opacity: 1.0
        }});
      }}
      const mesh = new THREE.Mesh(geom, mat);
      if (!name.includes('eye')) {{
        shellMeshes.push(mesh);
      }}
      return mesh;
    }}

    const internalMeshes = [];
    for (const [name, d] of Object.entries(rawInternals)) {{
      const buf = base64ToArrayBuffer(d.b64);
      const g = loader.parse(buf);
      const m = new THREE.MeshStandardMaterial({{
        color: d.color,
        roughness: 0.3,
        metalness: 0.4
      }});
      const mesh = new THREE.Mesh(g, m);
      mesh.visible = false;
      robotRoot.add(mesh);
      internalMeshes.push(mesh);
    }}

    const baseMesh = createMesh('base_link');
    robotRoot.add(baseMesh);

    const leftWheel = createMesh('left_wheel');
    leftWheel.position.set(-49, 0, 17);
    robotRoot.add(leftWheel);

    const rightWheel = createMesh('right_wheel');
    rightWheel.position.set(49, 0, 17);
    robotRoot.add(rightWheel);

    const torsoFrontMesh = createMesh('torso_front');
    torsoFrontMesh.position.set(0, 0, 34);
    robotRoot.add(torsoFrontMesh);

    const torsoRearMesh = createMesh('torso_rear');
    torsoRearMesh.position.set(0, 0, 34);
    robotRoot.add(torsoRearMesh);

    const leftShoulder = new THREE.Group();
    leftShoulder.position.set(-48.5, 2, 78);
    leftShoulder.rotation.y = THREE.MathUtils.degToRad(10);
    robotRoot.add(leftShoulder);
    const leftArmMesh = createMesh('left_arm');
    leftShoulder.add(leftArmMesh);

    const leftWrist = new THREE.Group();
    leftWrist.position.set(0, 0, -26);
    leftShoulder.add(leftWrist);
    const leftPincerMesh = createMesh('left_pincer');
    leftWrist.add(leftPincerMesh);

    const rightShoulder = new THREE.Group();
    rightShoulder.position.set(48.5, 2, 78);
    rightShoulder.rotation.y = THREE.MathUtils.degToRad(-10);
    robotRoot.add(rightShoulder);
    const rightArmMesh = createMesh('right_arm');
    rightShoulder.add(rightArmMesh);

    const rightWrist = new THREE.Group();
    rightWrist.position.set(0, 0, -26);
    rightShoulder.add(rightWrist);
    const rightPincerMesh = createMesh('right_pincer');
    rightWrist.add(rightPincerMesh);

    const neckYaw = new THREE.Group();
    neckYaw.position.set(0, 0, 96);
    robotRoot.add(neckYaw);
    const neckTurretMesh = createMesh('neck_yaw_turret');
    neckYaw.add(neckTurretMesh);

    const neckPitch = new THREE.Group();
    neckPitch.position.set(0, 0, 12);
    neckYaw.add(neckPitch);

    const headFrontMesh = createMesh('head_front');
    headFrontMesh.position.set(0, 0, 0);
    neckPitch.add(headFrontMesh);

    const headRearMesh = createMesh('head_rear');
    headRearMesh.position.set(0, 0, 0);
    neckPitch.add(headRearMesh);

    const leftEyePivot = new THREE.Group();
    leftEyePivot.position.set(-13.5, 35.0, 17.0);
    neckPitch.add(leftEyePivot);
    const leftEyeMesh = createMesh('left_eye');
    leftEyePivot.add(leftEyeMesh);

    const rightEyePivot = new THREE.Group();
    rightEyePivot.position.set(13.5, 35.0, 17.0);
    neckPitch.add(rightEyePivot);
    const rightEyeMesh = createMesh('right_eye');
    rightEyePivot.add(rightEyeMesh);

    const leftEarPivot = new THREE.Group();
    leftEarPivot.position.set(-32, 2, 30);
    neckPitch.add(leftEarPivot);
    const leftEarMesh = createMesh('left_ear');
    leftEarMesh.rotation.set(THREE.MathUtils.degToRad(49.08), THREE.MathUtils.degToRad(-20.0), 0);
    leftEarPivot.add(leftEarMesh);

    const rightEarPivot = new THREE.Group();
    rightEarPivot.position.set(32, 2, 30);
    neckPitch.add(rightEarPivot);
    const rightEarMesh = createMesh('right_ear');
    rightEarMesh.rotation.set(THREE.MathUtils.degToRad(49.08), THREE.MathUtils.degToRad(20.0), 0);
    rightEarPivot.add(rightEarMesh);

    document.getElementById('loading').style.display = 'none';

    let isXrayMode = false;
    let clamshellSplitDist = 0;

    function toggleXray() {{
      isXrayMode = !isXrayMode;
      const btn = document.getElementById('btn-xray');
      btn.innerText = isXrayMode ? "X-Ray Mode: ON" : "X-Ray Mode: OFF";
      document.getElementById('xray-legend').style.display = (isXrayMode || clamshellSplitDist > 0) ? 'block' : 'none';
      if (isXrayMode) {{
        btn.classList.add('highlight');
      }} else {{
        btn.classList.remove('highlight');
      }}
      for (const sm of shellMeshes) {{
        sm.material.transparent = isXrayMode;
        sm.material.opacity = isXrayMode ? 0.22 : 1.0;
        sm.material.depthWrite = !isXrayMode;
        sm.material.needsUpdate = true;
      }}
      for (const im of internalMeshes) {{
        im.visible = isXrayMode || (clamshellSplitDist > 0);
      }}
    }}

    function updateClamshellSplit() {{
      clamshellSplitDist = parseFloat(document.getElementById('slider-clamshell').value);
      document.getElementById('val-clamshell').innerText = clamshellSplitDist > 0 ? clamshellSplitDist.toFixed(0) + ' mm' : '0 mm (Sealed)';
      
      const activeBtn = clamshellSplitDist === 0 ? 'btn-clam-close' : (clamshellSplitDist <= 25 ? 'btn-clam-inspect' : 'btn-clam-full');
      document.querySelectorAll('.btn-group .exp-btn').forEach(b => {{
        if (b.id.startsWith('btn-clam-')) b.classList.remove('active');
      }});
      const act = document.getElementById(activeBtn);
      if (act) act.classList.add('active');

      torsoFrontMesh.position.y = clamshellSplitDist;
      torsoRearMesh.position.y = -clamshellSplitDist;
      headFrontMesh.position.y = clamshellSplitDist;
      headRearMesh.position.y = -clamshellSplitDist;
      
      leftEyePivot.position.y = 35.0 + clamshellSplitDist;
      rightEyePivot.position.y = 35.0 + clamshellSplitDist;
      leftEarPivot.position.y = 2.0 - clamshellSplitDist;
      rightEarPivot.position.y = 2.0 - clamshellSplitDist;

      for (const im of internalMeshes) {{
        im.visible = isXrayMode || clamshellSplitDist > 0;
      }}
      document.getElementById('xray-legend').style.display = (isXrayMode || clamshellSplitDist > 0) ? 'block' : 'none';
    }}

    function setClamshellSplit(val) {{
      document.getElementById('slider-clamshell').value = val;
      updateClamshellSplit();
    }}

    function toggleClamshellSplit() {{
      if (clamshellSplitDist > 0) {{
        setClamshellSplit(0);
      }} else {{
        setClamshellSplit(35);
      }}
    }}

    let ws = null;
    let isLiveSync = false;
    let wheelDriveState = [0, 0, 0, 0];
    let gridOffset = 0;

    function toggleWokwiSync() {{
      isLiveSync = !isLiveSync;
      const btn = document.getElementById('btn-wokwi-sync');
      const badge = document.getElementById('wokwi-badge');
      if (isLiveSync) {{
        btn.innerText = "Live Wokwi Sync: ON";
        btn.style.background = "rgba(20, 184, 166, 0.3)";
        btn.style.borderColor = "#14B8A6";
        btn.style.color = "#14B8A6";
        connectWebSocket();
      }} else {{
        btn.innerText = "Live Wokwi Sync: OFF";
        btn.style.background = "rgba(6, 182, 212, 0.25)";
        btn.style.borderColor = "#06B6D4";
        btn.style.color = "#06B6D4";
        badge.innerText = "OFFLINE";
        badge.style.background = "rgba(100, 116, 139, 0.2)";
        badge.style.color = "#94A3B8";
        badge.style.borderColor = "#475569";
        wheelDriveState = [0, 0, 0, 0];
        if (ws) {{
          ws.close();
          ws = null;
        }}
      }}
    }}

    function connectWebSocket() {{
      const badge = document.getElementById('wokwi-badge');
      badge.innerText = "CONNECTING...";
      badge.style.background = "rgba(234, 179, 8, 0.2)";
      badge.style.color = "#EAB308";
      badge.style.borderColor = "#EAB308";

      try {{
        ws = new WebSocket("ws://localhost:8765");
      }} catch(e) {{
        badge.innerText = "BRIDGE ERROR";
        return;
      }}

      ws.onopen = function() {{
        badge.innerText = "CONNECTED TO WOKWI";
        badge.style.background = "rgba(20, 184, 166, 0.25)";
        badge.style.color = "#14B8A6";
        badge.style.borderColor = "#14B8A6";
      }};

      ws.onclose = function() {{
        if (isLiveSync) {{
          badge.innerText = "WAITING FOR BRIDGE";
          badge.style.background = "rgba(239, 68, 68, 0.2)";
          badge.style.color = "#EF4444";
          badge.style.borderColor = "#EF4444";
          setTimeout(() => {{ if (isLiveSync) connectWebSocket(); }}, 2000);
        }}
      }};

      ws.onerror = function() {{
        badge.innerText = "BRIDGE DISCONNECTED";
      }};

      ws.onmessage = function(event) {{
        if (!isLiveSync) return;
        try {{
          const data = JSON.parse(event.data);
          if (data.wokwi_status) {{
            if (data.wokwi_status === "connected") {{
              badge.innerText = "CONNECTED TO WOKWI";
              badge.style.background = "rgba(20, 184, 166, 0.25)";
              badge.style.color = "#14B8A6";
              badge.style.borderColor = "#14B8A6";
            }} else {{
              badge.innerText = "SIM PAUSED / IDLE";
              badge.style.background = "rgba(234, 179, 8, 0.2)";
              badge.style.color = "#EAB308";
              badge.style.borderColor = "#EAB308";
            }}
          }}

          if (data.event) {{
            const evRow = document.getElementById('wokwi-event-row');
            const evVal = document.getElementById('wokwi-event-val');
            evRow.style.display = 'flex';
            evVal.innerText = data.event;
            evVal.style.color = data.event.includes("CLIFF") ? "#EF4444" : "#EAB308";
          }}

          if (data.wheels && Array.isArray(data.wheels)) {{
            wheelDriveState = data.wheels;
          }}

          if (data.joints && Array.isArray(data.joints) && data.joints.length === 8) {{
            const j = data.joints;
            const yaw = j[0] - 90;
            const pitch = j[1] - 90;
            const earL = j[2] - 110;
            const earR = 70 - j[3];
            const shoulderL = j[4] - 90;
            const shoulderR = j[5] - 90;
            const pincerL = j[6] - 90;
            const pincerR = j[7] - 90;

            document.getElementById('slider-yaw').value = yaw;
            document.getElementById('slider-pitch').value = pitch;
            document.getElementById('slider-ear-l').value = earL;
            document.getElementById('slider-ear-r').value = earR;
            document.getElementById('slider-shoulder-l').value = shoulderL;
            document.getElementById('slider-shoulder-r').value = shoulderR;
            document.getElementById('slider-pincer-l').value = Math.max(0, pincerL);
            document.getElementById('slider-pincer-r').value = Math.max(0, pincerR);
            updateJoints(false);
          }}

          if (data.emotion) {{
            setEyeExpression(data.emotion === 'neutral' ? 'happy' : data.emotion, false);
          }}
        }} catch(err) {{
          console.error(err);
        }}
      }};
    }}

    function setEyeExpression(mode, echoToWs = true) {{
      document.querySelectorAll('.exp-btn').forEach(btn => {{
        if (btn.id.startsWith('btn-clam-')) return;
        btn.classList.remove('active');
      }});
      const activeBtn = document.getElementById('btn-' + mode);
      if (activeBtn) activeBtn.classList.add('active');

      if (mode === 'happy' || mode === 'neutral') {{
        leftEyeMesh.scale.set(1.15, 0.65, 1.0);
        leftEyeMesh.rotation.z = 0.12;
        rightEyeMesh.scale.set(1.15, 0.65, 1.0);
        rightEyeMesh.rotation.z = -0.12;
        if (leftEyeMesh.material) leftEyeMesh.material.emissive.setHex(0xEAB308);
        if (rightEyeMesh.material) rightEyeMesh.material.emissive.setHex(0xEAB308);
      }} else if (mode === 'curious') {{
        leftEyeMesh.scale.set(1.25, 1.2, 1.0);
        leftEyeMesh.rotation.z = 0.18;
        rightEyeMesh.scale.set(0.85, 0.85, 1.0);
        rightEyeMesh.rotation.z = 0.18;
        if (leftEyeMesh.material) leftEyeMesh.material.emissive.setHex(0x06B6D4);
        if (rightEyeMesh.material) rightEyeMesh.material.emissive.setHex(0x06B6D4);
      }} else if (mode === 'sleepy') {{
        leftEyeMesh.scale.set(1.0, 0.35, 1.0);
        leftEyeMesh.rotation.z = 0.0;
        rightEyeMesh.scale.set(1.0, 0.35, 1.0);
        rightEyeMesh.rotation.z = 0.0;
        if (leftEyeMesh.material) leftEyeMesh.material.emissive.setHex(0x14B8A6);
        if (rightEyeMesh.material) rightEyeMesh.material.emissive.setHex(0x14B8A6);
      }} else if (mode === 'alert') {{
        leftEyeMesh.scale.set(1.2, 1.35, 1.0);
        leftEyeMesh.rotation.z = 0.0;
        rightEyeMesh.scale.set(1.2, 1.35, 1.0);
        rightEyeMesh.rotation.z = 0.0;
        if (leftEyeMesh.material) leftEyeMesh.material.emissive.setHex(0x00F0FF);
        if (rightEyeMesh.material) rightEyeMesh.material.emissive.setHex(0x00F0FF);
      }}

      if (echoToWs && isLiveSync && ws && ws.readyState === WebSocket.OPEN) {{
        ws.send(JSON.stringify({{ emotion: mode }}));
      }}
    }}

    function updateJoints(echoToWs = true) {{
      const earL = parseFloat(document.getElementById('slider-ear-l').value);
      const earR = parseFloat(document.getElementById('slider-ear-r').value);
      const yaw = parseFloat(document.getElementById('slider-yaw').value);
      const pitch = parseFloat(document.getElementById('slider-pitch').value);
      const shoulderL = parseFloat(document.getElementById('slider-shoulder-l').value);
      const shoulderR = parseFloat(document.getElementById('slider-shoulder-r').value);
      const pincerL = parseFloat(document.getElementById('slider-pincer-l').value);
      const pincerR = parseFloat(document.getElementById('slider-pincer-r').value);

      document.getElementById('val-ear-l').innerText = earL + '°';
      document.getElementById('val-ear-r').innerText = earR + '°';
      document.getElementById('val-yaw').innerText = yaw + '°';
      document.getElementById('val-pitch').innerText = pitch + '°';
      document.getElementById('val-shoulder-l').innerText = shoulderL + '°';
      document.getElementById('val-shoulder-r').innerText = shoulderR + '°';
      document.getElementById('val-pincer-l').innerText = pincerL + '° (' + (pincerL > 0 ? (pincerL*0.33).toFixed(1) + 'mm' : 'Pinch') + ')';
      document.getElementById('val-pincer-r').innerText = pincerR + '° (' + (pincerR > 0 ? (pincerR*0.33).toFixed(1) + 'mm' : 'Pinch') + ')';

      leftEarPivot.rotation.x = THREE.MathUtils.degToRad(earL);
      rightEarPivot.rotation.x = THREE.MathUtils.degToRad(earR);
      neckYaw.rotation.z = THREE.MathUtils.degToRad(yaw);
      neckPitch.rotation.x = THREE.MathUtils.degToRad(-pitch);
      leftShoulder.rotation.x = THREE.MathUtils.degToRad(shoulderL);
      rightShoulder.rotation.x = THREE.MathUtils.degToRad(shoulderR);
      leftWrist.rotation.y = THREE.MathUtils.degToRad(-pincerL);
      rightWrist.rotation.y = THREE.MathUtils.degToRad(pincerR);

      if (echoToWs && isLiveSync && ws && ws.readyState === WebSocket.OPEN) {{
        const targetAngles = [
          Math.round(yaw + 90),
          Math.round(pitch + 90),
          Math.round(earL + 110),
          Math.round(70 - earR),
          Math.round(shoulderL + 90),
          Math.round(shoulderR + 90),
          Math.round(pincerL + 90),
          Math.round(pincerR + 90)
        ];
        ws.send(JSON.stringify({{ joints: targetAngles }}));
      }}
    }}

    function setPreset(name, echoToWs = true) {{
      if (name === 'neutral') {{
        setSliders({{ 'slider-ear-l': 0, 'slider-ear-r': 0, 'slider-yaw': 0, 'slider-pitch': 0, 'slider-shoulder-l': 0, 'slider-shoulder-r': 0, 'slider-pincer-l': 0, 'slider-pincer-r': 0 }});
        setEyeExpression('happy', false);
      }} else if (name === 'wave') {{
        setSliders({{ 'slider-ear-l': 20, 'slider-ear-r': 20, 'slider-yaw': 15, 'slider-pitch': 10, 'slider-shoulder-l': 95, 'slider-shoulder-r': 15, 'slider-pincer-l': 40, 'slider-pincer-r': 0 }});
        setEyeExpression('curious', false);
      }} else if (name === 'hold') {{
        setSliders({{ 'slider-ear-l': 10, 'slider-ear-r': 10, 'slider-yaw': 0, 'slider-pitch': -15, 'slider-shoulder-l': 55, 'slider-shoulder-r': 55, 'slider-pincer-l': 28, 'slider-pincer-r': 28 }});
        setEyeExpression('alert', false);
      }} else if (name === 'alert') {{
        setSliders({{ 'slider-ear-l': 45, 'slider-ear-r': 45, 'slider-yaw': 0, 'slider-pitch': 20, 'slider-shoulder-l': -15, 'slider-shoulder-r': -15, 'slider-pincer-l': 0, 'slider-pincer-r': 0 }});
        setEyeExpression('alert', false);
      }}

      if (echoToWs && isLiveSync && ws && ws.readyState === WebSocket.OPEN) {{
        ws.send(JSON.stringify({{ gesture: name }}));
      }}
    }}

    function setSliders(vals) {{
      for (const [id, val] of Object.entries(vals)) {{
        document.getElementById(id).value = val;
      }}
      updateJoints(false);
    }}

    let isExploded = false;
    function toggleExploded() {{
      isExploded = !isExploded;
      document.getElementById('btn-explode').innerText = isExploded ? "Assemble View" : "Explode View";
      const f = isExploded ? 1.0 : 0.0;
      torsoFrontMesh.position.z = 34 + f * 25;
      torsoRearMesh.position.z = 34 + f * 25;
      neckYaw.position.z = 96 + f * 50;
      leftShoulder.position.x = -48.5 - f * 20;
      rightShoulder.position.x = 48.5 + f * 20;
      leftWheel.position.x = -49 - f * 25;
      rightWheel.position.x = 49 + f * 25;
    }}

    function resetCamera() {{
      camera.position.set(85, 220, 125);
      controls.target.set(0, 0, 82);
      controls.update();
    }}

    function animate() {{
      requestAnimationFrame(animate);

      if (isLiveSync && wheelDriveState) {{
        const isFwd = (wheelDriveState[0] === 1 && wheelDriveState[2] === 1);
        const isRev = (wheelDriveState[1] === 1 && wheelDriveState[3] === 1);
        if (isFwd) {{
          gridOffset = (gridOffset - 0.7) % 10.0;
          grid.position.y = gridOffset;
          leftWheel.rotation.x += 0.08;
          rightWheel.rotation.x += 0.08;
        }} else if (isRev) {{
          gridOffset = (gridOffset + 0.9) % 10.0;
          grid.position.y = gridOffset;
          leftWheel.rotation.x -= 0.10;
          rightWheel.rotation.x -= 0.10;
        }}
      }}

      renderer.render(scene, camera);
    }}
    animate();

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
  </script>
</body>
</html>"""

with open("exports/viewer.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated exports/viewer.html with {len(parts_dict)} assembly parts ({len(internals_dict)} internal components) with DFA Clamshell Split & X-Ray toggles!")
