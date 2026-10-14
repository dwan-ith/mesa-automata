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
                self.width = width
            if height:
                self.height = height

            pattern_info = PATTERNS.get(self.pattern_key)
            w = self.width
            h = self.height
            if pattern_info:
                w = max(w, pattern_info.get("width", 35))
                h = max(h, pattern_info.get("height", 35))

            self.width = w
            self.height = h

            pat = get_pattern(self.pattern_key) if self.pattern_key in PATTERNS else {}
            self.model = VonNeumannCA(width=w, height=h, initial_pattern=pat)

    def step(self, n: int = 1):
        with self.lock:
            if self.model:
                for _ in range(max(1, n)):
                    self.model.step()

    def set_cell(self, x: int, y: int, state: int):
        with self.lock:
            if self.model:
                self.model.set_cell(x, y, state)

    def load_sequence_arm(self, sequence: str):
        with self.lock:
            pat = make_programmable_builder(sequence, x=8, y=self.height // 2)
            self.model = VonNeumannCA(width=max(self.width, len(sequence) + 20), height=self.height, initial_pattern=pat)
            self.pattern_key = f"Custom Sequence: {sequence}"

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
            SIMULATION.reset(pattern)
            self.send_json(SIMULATION.get_snapshot())
        elif url.path == "/api/set_cell":
            x = int(payload.get("x", 0))
            y = int(payload.get("y", 0))
            state = int(payload.get("state", 0))
            SIMULATION.set_cell(x, y, state)
            self.send_json(SIMULATION.get_snapshot())
        elif url.path == "/api/sequence":
            seq = payload.get("sequence", "1011").strip()
            SIMULATION.load_sequence_arm(seq)
            self.send_json(SIMULATION.get_snapshot())
        else:
            self.send_error(404, "Unknown API endpoint")

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
