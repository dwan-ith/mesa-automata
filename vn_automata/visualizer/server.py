"""Lightweight HTTP server providing interactive web visualization for Von Neumann CA."""

import http.server
import json
import os
import socketserver
import threading
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any

from ..states import VNState, STATE_INFO, STATE_COLORS, StateCategory
from ..model import VonNeumannCA
from ..patterns import PATTERNS, get_pattern, make_programmable_builder


class SimulationState:
    """Thread-safe simulation manager."""

    def __init__(self, pattern_key: str = "construction_arm_1011", width: int = 40, height: int = 30):
        self.lock = threading.Lock()
        self.pattern_key = pattern_key
        self.width = width
        self.height = height
        self.model: Optional[VonNeumannCA] = None
        self.reset(pattern_key)

    def reset(self, pattern_key: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None):
        with self.lock:
            if pattern_key:
                self.pattern_key = pattern_key
            if width:
                self.width = max(5, min(int(width), 200))
            if height:
                self.height = max(5, min(int(height), 200))

            pat = get_pattern(self.pattern_key) if self.pattern_key in PATTERNS else {}
            # Expand only if the pattern itself would be clipped, never force
            # a minimum size on the user.
            if pat:
                need_w = max(px for px, _ in pat) + 2
                need_h = max(py for _, py in pat) + 2
                self.width = max(self.width, min(need_w, 200))
                self.height = max(self.height, min(need_h, 200))

            self.model = VonNeumannCA(width=self.width, height=self.height, initial_pattern=pat)

    def step(self, n: int = 1):
        with self.lock:
            if self.model:
                for _ in range(max(1, min(int(n), 1000))):
                    self.model.step()

    def set_cell(self, x: int, y: int, state: int):
        with self.lock:
            if self.model:
                if not (0 <= int(state) <= 28):
                    raise ValueError(f"State must be 0..28, got {state}")
                self.model.set_cell(int(x), int(y), int(state))

    def load_sequence_arm(self, sequence: str):
        seq = (sequence or "").strip()
        if not seq or any(b not in ("0", "1") for b in seq):
            raise ValueError("Sequence must be binary ('0'/'1' only)")
        if len(seq) > 32:
            raise ValueError("Sequence too long (max 32 bits)")
        with self.lock:
            pat = make_programmable_builder(seq, x=8, y=self.height // 2)
            self.model = VonNeumannCA(width=max(self.width, len(seq) + 20), height=self.height, initial_pattern=pat)
            self.pattern_key = f"Custom Sequence: {seq}"

    def get_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            if not self.model:
                return {}

            # Non-ground cells as sparse map: {"x,y": state}
            cells = {}
            for (x, y), agent in self.model.cell_map.items():
                if agent.state != VNState.U:
                    cells[f"{x},{y}"] = agent.state

            return {
                "step": self.model.step_count,
                "width": self.model.width,
                "height": self.model.height,
                "pattern": self.pattern_key,
                "cells": cells,
                "metrics": {
                    "active_pulses": self.model.count_active_pulses(),
                    "non_ground": self.model.count_non_ground_cells(),
                    "sensitized": self.model.count_sensitized_cells(),
                    "confluent": self.model.count_confluent_cells(),
                }
            }


SIMULATION = SimulationState()


class VNApiHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler serving static visualizer files and simulation REST API."""

    def __init__(self, *args, **kwargs):
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        super().__init__(*args, directory=static_dir, **kwargs)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/snapshot":
            self.send_json(SIMULATION.get_snapshot())
        elif url.path == "/api/patterns":
            pat_list = [
                {"key": k, "name": v["name"], "description": v["description"]}
                for k, v in PATTERNS.items()
            ]
            self.send_json(pat_list)
        elif url.path == "/api/states":
            states_data = []
            for state in VNState:
                meta = STATE_INFO[state]
                states_data.append({
                    "code": meta.code,
                    "name": meta.name,
                    "symbol": meta.symbol,
                    "category": meta.category.name,
                    "category_id": int(meta.category),
                    "is_excited": meta.is_excited,
                    "color": meta.hex_color,
                    "description": meta.description,
                })
            self.send_json(states_data)
        else:
            super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        if url.path == "/api/step":
            count = int(payload.get("count", 1))
            SIMULATION.step(count)
            self.send_json(SIMULATION.get_snapshot())
        elif url.path == "/api/reset":
            pattern = payload.get("pattern", "construction_arm_1011")
            if pattern not in PATTERNS:
                self.send_error_json(400, f"Unknown pattern {pattern!r}")
                return
            SIMULATION.reset(pattern)
            self.send_json(SIMULATION.get_snapshot())
        elif url.path == "/api/set_cell":
            try:
                x = int(payload.get("x", 0))
                y = int(payload.get("y", 0))
                state = int(payload.get("state", 0))
                SIMULATION.set_cell(x, y, state)
            except (ValueError, TypeError) as exc:
                self.send_error_json(400, str(exc))
                return
            self.send_json(SIMULATION.get_snapshot())
        elif url.path == "/api/sequence":
            seq = str(payload.get("sequence", "1011")).strip()
            try:
                SIMULATION.load_sequence_arm(seq)
            except (ValueError, TypeError) as exc:
                self.send_error_json(400, str(exc))
                return
            self.send_json(SIMULATION.get_snapshot())
        else:
            self.send_error(404, "Unknown API endpoint")

    def send_error_json(self, code: int, message: str):
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data: Any):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Silence verbose request logs
        return


def run_server(port: int = 8521, pattern: str = "construction_arm_1011"):
    """Start visualizer HTTP server."""
    SIMULATION.reset(pattern)
    with socketserver.TCPServer(("", port), VNApiHandler) as httpd:
        print(f"\n=======================================================")
        print(f"  Mesa Von Neumann CA Visualizer running at:")
        print(f"  http://localhost:{port}")
        print(f"=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
