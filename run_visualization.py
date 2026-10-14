#!/usr/bin/env python3
"""CLI runner to launch the interactive Von Neumann 29-State CA visualizer."""

import argparse
import webbrowser
from vn_automata.visualizer import run_server
from vn_automata.patterns import PATTERNS


def main():
    parser = argparse.ArgumentParser(
        description="Launch Von Neumann 29-State Self-Reproducing Automaton Visualizer"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8521,
        help="Port to serve the interactive web interface (default: 8521)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="construction_arm_1011",
        choices=list(PATTERNS.keys()),
        help="Initial pattern to load into the simulation",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open browser",
    )

    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        # Schedule browser opening
        import threading
        import time

        def open_tab():
            time.sleep(0.8)
            webbrowser.open(url)

        threading.Thread(target=open_tab, daemon=True).start()

    run_server(port=args.port, pattern=args.pattern)


if __name__ == "__main__":
    main()
