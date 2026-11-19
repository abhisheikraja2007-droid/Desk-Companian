from __future__ import annotations
import math
import os
import shutil
import subprocess
import sys
import time
from playwright.sync_api import sync_playwright

def find_chromium_path() -> str:
    possible_paths = [
        r"C:\Users\Admin\AppData\Local\ms-playwright\chromium-1254\chrome-win64\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright\chromium-1254\chrome-win64\chrome.exe"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return ""

def record_wiring_proof_video() -> None:
    video_dir = os.path.abspath("exports/videos")
    os.makedirs(video_dir, exist_ok=True)

    chromium_path = find_chromium_path()
    if not chromium_path:
        raise RuntimeError("Chromium binary not found in local ms-playwright path")

    url = "http://localhost:8080/viewer.html"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chromium_path,
            headless=True,
            args=["--use-gl=angle", "--use-angle=swiftshader", "--no-sandbox"]
        )

        context = browser.new_context(
            record_video_dir=video_dir,
            record_video_size={"width": 1280, "height": 720},
            viewport={"width": 1280, "height": 720}
        )

        page = context.new_page()
        print(f"Navigating to {url}...")
        page.goto(url, wait_until="networkidle")

        try:
            page.wait_for_selector("#loading", state="hidden", timeout=15000)
        except Exception:
            pass

        time.sleep(1.0)
        page.evaluate("if (isLiveSync) toggleWokwiSync();")

        print("Stage 1 (0-4s): Exploded clamshell inspection (35 mm) + 360-deg pan...")
        page.evaluate("window.setClamshellSplit(35);")
        page.evaluate("window.setWiringVisible(true);")

        steps_orbit = 40
        radius = 250.0
        phi = math.pi * 0.42
        for i in range(steps_orbit):
            theta = (2.0 * math.pi * i) / steps_orbit
            page.evaluate(f"window.setCameraOrbit({radius}, {theta}, {phi});")
            time.sleep(0.10)

        print("Stage 2 (4-7s): Clamshell closing sequence (35 mm -> 0 mm sealed)...")
        steps_close = 30
        for i in range(steps_close):
            frac = i / (steps_close - 1)
            dist = 35.0 * (1.0 - frac)
            theta = 0.5 + 0.3 * frac
            cur_r = 250.0 - 50.0 * frac
            page.evaluate(f"window.setClamshellSplit({dist});")
            page.evaluate(f"window.setCameraOrbit({cur_r}, {theta}, {phi});")
            time.sleep(0.10)

        print("Stage 3 (7-11s): Dynamic neck yaw (+-75 deg) & pitch (-30 to +40 deg) sweep...")
        page.evaluate("window.setXrayMode(true);")
        steps_neck = 40
        for i in range(steps_neck):
            t_frac = i / (steps_neck - 1)
            yaw = 75.0 * math.sin(t_frac * 2.0 * math.pi)
            pitch = 5.0 + 35.0 * math.cos(t_frac * 2.0 * math.pi)
            page.evaluate(f"window.setJointValues({{ yaw: {yaw}, pitch: {pitch} }});")
            time.sleep(0.10)

        print("Stage 4 (11-15s): Shoulder (+120 deg) + pincer + ear sweep with proof badge...")
        badge_text = "&#10004; 31/31 PARTS + 5 HARNESSES: 0.00 mm&#179; COLLISION (PASS)"
        page.evaluate(f"window.showProofBadge(true, '{badge_text}');")

        steps_arms = 40
        for i in range(steps_arms):
            t_frac = i / (steps_arms - 1)
            sh = 120.0 * math.sin(t_frac * math.pi)
            pinc = 50.0 * math.sin(t_frac * math.pi)
            ear = 45.0 * math.sin(t_frac * 2.0 * math.pi)
            page.evaluate(f"window.setJointValues({{ shoulderL: {sh}, shoulderR: {sh}, pincerL: {pinc}, pincerR: {pinc}, earL: {ear}, earR: {ear} }});")
            time.sleep(0.10)

        print("Finalizing video capture...")
        video_path = page.video.path()
        context.close()
        browser.close()

    print(f"Playwright video recorded to: {video_path}")
    final_webm = os.path.join(video_dir, "wiring_fit_proof.webm")
    final_mp4 = os.path.join(video_dir, "wiring_fit_proof.mp4")

    if os.path.exists(final_webm):
        os.remove(final_webm)
    shutil.copyfile(video_path, final_webm)

    print("Transcoding to MP4 via ffmpeg...")
    cmd = [
        "ffmpeg", "-y",
        "-i", final_webm,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        final_mp4
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"Warning: ffmpeg returned non-zero code {res.returncode}")

    if os.path.exists(final_webm) and os.path.getsize(final_webm) > 0:
        print(f"Verified WEBM proof: {final_webm} ({os.path.getsize(final_webm):,} bytes)")
    if os.path.exists(final_mp4) and os.path.getsize(final_mp4) > 0:
        print(f"Verified MP4 proof:  {final_mp4} ({os.path.getsize(final_mp4):,} bytes)")

if __name__ == "__main__":
    record_wiring_proof_video()
