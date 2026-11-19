import asyncio
import json
import queue
import sys
import threading
import time
from typing import Set

import serial
from websockets.asyncio.server import ServerConnection, serve

RFC2217_URL = "rfc2217://localhost:4000"
WS_HOST = "0.0.0.0"
WS_PORT = 8765

connected_clients: Set[ServerConnection] = set()
to_wokwi_queue: queue.Queue[str] = queue.Queue()
from_wokwi_queue: queue.Queue[str] = queue.Queue()
serial_connected = False

def serial_worker() -> None:
    global serial_connected
    while True:
        try:
            ser = serial.serial_for_url(RFC2217_URL, baudrate=115200, timeout=0.1)
            serial_connected = True
            from_wokwi_queue.put(json.dumps({"wokwi_status": "connected"}))
            print(f"[Bridge] Connected to Wokwi RFC2217 at {RFC2217_URL}")

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
                                from_wokwi_queue.put(raw_line)
                            except json.JSONDecodeError:
                                from_wokwi_queue.put(json.dumps({"event": raw_line}))
                        else:
                            from_wokwi_queue.put(json.dumps({"event": raw_line}))
                else:
                    time.sleep(0.005)

        except Exception as e:
            if serial_connected:
                print(f"[Bridge] Disconnected from Wokwi: {e}")
            serial_connected = False
            from_wokwi_queue.put(json.dumps({"wokwi_status": "disconnected"}))
            time.sleep(1.5)

async def ws_handler(websocket: ServerConnection) -> None:
    connected_clients.add(websocket)
    status_msg = json.dumps({"wokwi_status": "connected" if serial_connected else "disconnected"})
    await websocket.send(status_msg)
    try:
        async for message in websocket:
            to_wokwi_queue.put(str(message))
    except Exception:
        pass
    finally:
        connected_clients.discard(websocket)

async def broadcast_loop() -> None:
    while True:
        msgs = []
        try:
            while not from_wokwi_queue.empty():
                msgs.append(from_wokwi_queue.get_nowait())
        except queue.Empty:
            pass

        if msgs and connected_clients:
            for msg in msgs:
                dead_clients = set()
                for client in connected_clients:
                    try:
                        await client.send(msg)
                    except Exception:
                        dead_clients.add(client)
                connected_clients.difference_update(dead_clients)

        await asyncio.sleep(0.01)

async def main() -> None:
    t = threading.Thread(target=serial_worker, daemon=True)
    t.start()

    server = await serve(ws_handler, WS_HOST, WS_PORT)
    print(f"[Bridge] WebSocket Telemetry Server running on ws://localhost:{WS_PORT}")
    
    await broadcast_loop()
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
