import argparse
import asyncio
import functools
import http.server
import json
import math
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Set

import serial
from playwright.async_api import async_playwright
from websockets.asyncio.server import ServerConnection, serve

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
EXPORTS_DIR = WORKSPACE_DIR / "exports"
VIDEOS_DIR = EXPORTS_DIR / "videos"
RFC2217_URL = "rfc2217://localhost:4000"

connected_clients: Set[ServerConnection] = set()
broadcast_queue: queue.Queue[str] = queue.Queue()
to_wokwi_queue: queue.Queue[str] = queue.Queue()
wokwi_active = False

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

def start_http_server(port: int) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(EXPORTS_DIR))
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd

def rfc2217_worker() -> None:
    global wokwi_active
    while True:
        try:
            ser = serial.serial_for_url(RFC2217_URL, baudrate=115200, timeout=0.1)
            wokwi_active = True
            broadcast_queue.put(json.dumps({"wokwi_status": "connected", "mode": "WOKWI LIVE (PORT 4000)"}))
            print(f"[Bridge] Connected to live Wokwi RFC2217 socket on {RFC2217_URL}")

            while True:
                try:
                    while not to_wokwi_queue.empty():
                        cmd = to_wokwi_queue.get_nowait()
                        if not cmd.endswith("\n"):
                            cmd += "\n"
                        ser.write(cmd.encode("utf-8"))
                        ser.flush()
                except queue.Empty:
                    pass

                line_bytes = ser.readline()
                if line_bytes:
                    raw_line = line_bytes.decode("utf-8", errors="ignore").strip()
                    if raw_line:
                        if raw_line.startswith("{") and raw_line.endswith("}"):
                            try:
                                json.loads(raw_line)
                                broadcast_queue.put(raw_line)
                            except json.JSONDecodeError:
                                broadcast_queue.put(json.dumps({"event": raw_line}))
                        else:
                            broadcast_queue.put(json.dumps({"event": raw_line}))
                else:
                    time.sleep(0.005)

        except Exception:
            wokwi_active = False
            time.sleep(1.5)

async def ws_handler(websocket: ServerConnection) -> None:
    connected_clients.add(websocket)
    status_msg = json.dumps({
        "wokwi_status": "connected" if wokwi_active else "simulated",
        "mode": "WOKWI LIVE (PORT 4000)" if wokwi_active else "SIMULATED DEMO (20 HZ)"
    })
    await websocket.send(status_msg)
    try:
        async for message in websocket:
            to_wokwi_queue.put(str(message))
    except Exception:
        pass
    finally:
        connected_clients.discard(websocket)

async def websocket_broadcaster() -> None:
    while True:
        msgs = []
        try:
            while not broadcast_queue.empty():
                msgs.append(broadcast_queue.get_nowait())
        except queue.Empty:
            pass

        if msgs and connected_clients:
            for msg in msgs:
                dead = set()
                for client in connected_clients:
                    try:
                        await client.send(msg)
                    except Exception:
                        dead.add(client)
                connected_clients.difference_update(dead)

        await asyncio.sleep(0.01)

def compute_choreography_frame(t: float) -> dict:
    frame = {
        "t": int(t * 1000),
        "mode": "WOKWI LIVE" if wokwi_active else "SIMULATED DEMO (20 Hz)"
    }

    if t < 3.0:
        p = t / 3.0
        yaw = 90.0 + 25.0 * math.sin(2.0 * math.pi * p * 2.0)
        frame.update({
            "state": "roam",
            "emotion": "neutral",
            "wheels": [1, 0, 1, 0],
            "cliff": [0, 0, 0, 0],
            "joints": [round(yaw, 1), 90.0, 110.0, 70.0, 90.0, 90.0, 90.0, 90.0]
        })
    elif t < 6.0:
        p = (t - 3.0) / 3.0
        alpha = 0.5 * (1.0 - math.cos(math.pi * min(1.0, p * 2.5)))
        arm_l = 90.0 + (150.0 - 90.0) * alpha
        pin_l = 90.0 + (135.0 - 90.0) * alpha
        wiggle = 20.0 * math.sin(4.0 * math.pi * p * 2.0)
        ear_l = 110.0 + wiggle
        ear_r = 70.0 - wiggle
        frame.update({
            "state": "petted",
            "emotion": "happy",
            "event": "[TOUCH] Crown petted!",
            "wheels": [0, 0, 0, 0],
            "cliff": [0, 0, 0, 0],
            "joints": [90.0, 85.0, round(ear_l, 1), round(ear_r, 1), round(arm_l, 1), 90.0, round(pin_l, 1), 90.0]
        })
    elif t < 9.0:
        p = (t - 6.0) / 3.0
        is_recoil = p > 0.05
        frame.update({
            "state": "cliff_recovery",
            "emotion": "alert",
            "event": "[CLIFF] Triggered sensor 1! Reversing...",
            "wheels": [0, 1, 0, 1] if is_recoil else [0, 0, 0, 0],
            "cliff": [1, 0, 0, 0],
            "joints": [90.0, 110.0, 155.0, 25.0, 100.0, 80.0, 90.0, 90.0]
        })
    elif t < 12.0:
        p = (t - 9.0) / 3.0
        alpha = 0.5 * (1.0 - math.cos(math.pi * min(1.0, p * 2.0)))
        yaw = 90.0 + (115.0 - 90.0) * alpha
        pitch = 90.0 + (105.0 - 90.0) * alpha
        ear_l = 110.0 + (135.0 - 110.0) * alpha
        ear_r = 70.0 + (75.0 - 70.0) * alpha
        arm_l = 90.0 + (110.0 - 90.0) * alpha
        arm_r = 90.0 + (70.0 - 90.0) * alpha
        frame.update({
            "state": "inspect",
            "emotion": "curious",
            "wheels": [0, 0, 0, 0],
            "cliff": [0, 0, 0, 0],
            "joints": [round(yaw, 1), round(pitch, 1), round(ear_l, 1), round(ear_r, 1), round(arm_l, 1), round(arm_r, 1), 90.0, 90.0]
        })
    else:
        p = (t - 12.0) / 3.0
        alpha = 0.5 * (1.0 - math.cos(math.pi * min(1.0, p * 2.0)))
        arm_l = 110.0 + (135.0 - 110.0) * alpha
        arm_r = 70.0 + (135.0 - 70.0) * alpha
        pin_l = 90.0 + (120.0 - 90.0) * alpha
        pin_r = 90.0 + (120.0 - 90.0) * alpha
        yaw = 115.0 + (90.0 - 115.0) * alpha
        pitch = 105.0 + (90.0 - 105.0) * alpha
        ear_l = 135.0 + (110.0 - 135.0) * alpha
        ear_r = 75.0 + (70.0 - 75.0) * alpha
        frame.update({
            "state": "hold_pen",
            "emotion": "curious",
            "wheels": [0, 0, 0, 0],
            "cliff": [0, 0, 0, 0],
            "joints": [round(yaw, 1), round(pitch, 1), round(ear_l, 1), round(ear_r, 1), round(arm_l, 1), round(arm_r, 1), round(pin_l, 1), round(pin_r, 1)]
        })

    return frame

async def run_choreography_loop(duration: float) -> None:
    steps = int(duration * 20)
    for i in range(steps):
        t = i * 0.05
        frame = compute_choreography_frame(t)
        broadcast_queue.put(json.dumps(frame))
        await asyncio.sleep(0.05)

def get_chromium_exec_path() -> Optional[str]:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = [
        local_app_data / "ms-playwright" / "chromium-1254" / "chrome-win64" / "chrome.exe",
        local_app_data / "ms-playwright" / "chromium-1248" / "chrome-win64" / "chrome.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None

async def record_playwright_video(port: int, duration: float, headless: bool) -> Path:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    temp_video_dir = VIDEOS_DIR / "temp_record"
    if temp_video_dir.exists():
        shutil.rmtree(temp_video_dir)
    temp_video_dir.mkdir(parents=True, exist_ok=True)

    exec_path = get_chromium_exec_path()

    print(f"[Recorder] Launching Playwright Chromium (executable: {exec_path or 'default'})...")
    async with async_playwright() as p:
        launch_kwargs = {"headless": headless}
        if exec_path:
            launch_kwargs["executable_path"] = exec_path
        
        browser = await p.chromium.launch(**launch_kwargs)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(temp_video_dir),
            record_video_size={"width": 1280, "height": 720}
        )
        page = await context.new_page()

        target_url = f"http://localhost:{port}/viewer.html"
        print(f"[Recorder] Navigating to {target_url}...")
        await page.goto(target_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)

        print(f"[Recorder] Recording 15s live 5-phase choreography...")
        choreo_task = asyncio.create_task(run_choreography_loop(duration))
        await choreo_task
        await page.wait_for_timeout(500)

        video_obj = page.video
        await page.close()
        await context.close()
        await browser.close()

        raw_video_path = await video_obj.path() if video_obj else None
        if not raw_video_path or not Path(raw_video_path).exists():
            found = list(temp_video_dir.glob("*.webm"))
            if found:
                raw_video_path = str(found[0])

    target_webm = VIDEOS_DIR / "desk_pet_live_demo.webm"
    if raw_video_path and Path(raw_video_path).exists():
        shutil.move(raw_video_path, str(target_webm))
        if temp_video_dir.exists():
            shutil.rmtree(temp_video_dir, ignore_errors=True)

    return target_webm

def convert_webm_to_mp4(webm_path: Path) -> Optional[Path]:
    mp4_path = webm_path.with_suffix(".mp4")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(webm_path),
            "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(mp4_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 0:
            return mp4_path
    except Exception as e:
        print(f"[FFmpeg] Conversion warning: {e}")
    return None

async def main_async(args: argparse.Namespace) -> None:
    httpd = start_http_server(args.port)
    print(f"[HTTP] Serving {EXPORTS_DIR} on http://localhost:{args.port}/viewer.html")

    serial_thread = threading.Thread(target=rfc2217_worker, daemon=True)
    serial_thread.start()

    ws_server = await serve(ws_handler, "0.0.0.0", args.ws_port)
    print(f"[WebSocket] Telemetry Server active on ws://localhost:{args.ws_port}")

    broadcaster_task = asyncio.create_task(websocket_broadcaster())

    target_webm = None
    target_mp4 = None
    if not args.skip_record:
        target_webm = await record_playwright_video(args.port, args.duration, not args.no_headless)
        webm_size_mb = target_webm.stat().st_size / (1024 * 1024) if target_webm.exists() else 0.0
        print(f"[Recorder] Saved WebM: {target_webm} ({webm_size_mb:.2f} MB)")

        target_mp4 = convert_webm_to_mp4(target_webm)
        if target_mp4:
            mp4_size_mb = target_mp4.stat().st_size / (1024 * 1024)
            print(f"[Recorder] Converted MP4: {target_mp4} ({mp4_size_mb:.2f} MB)")

    print("\n" + "=" * 90)
    print("                     RECORDING & 3D VERIFICATION SUMMARY")
    print("=" * 90)
    if target_webm:
        print(f"WebM Video:  {target_webm.resolve()}")
    if target_mp4:
        print(f"MP4 Video:   {target_mp4.resolve()}")
    print(f"Web Viewer:  http://localhost:{args.port}/viewer.html")
    print(f"WS Bridge:   ws://localhost:{args.ws_port}")
    print("=" * 90 + "\n")

    if args.live:
        print("[Live Mode] HTTP and WebSocket servers running indefinitely. Press Ctrl+C to terminate.")
        await asyncio.Future()

def main() -> None:
    parser = argparse.ArgumentParser(description="Desk Pet Live 3D Bridge & Playwright Video Recorder")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port (default: 8080)")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port (default: 8765)")
    parser.add_argument("--duration", type=float, default=15.0, help="Demo duration in seconds (default: 15.0)")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window while recording")
    parser.add_argument("--skip-record", action="store_true", help="Skip recording and launch live servers directly")
    parser.add_argument("--live", action="store_true", help="Keep servers running after recording")
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[Bridge] Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
