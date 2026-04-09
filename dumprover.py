from __future__ import annotations

import argparse
import json

from dumbdrive import REMOTE_SERVER_URL, make_sanitized_telemetry
from main import LIDAR_SENSOR_LABELS, parse_pose
from rover_control import close_rover_socket, configure_remote_server, fetch_rover_json, open_rover_socket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump current rover position and lidar readings.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use the local UDP TSS connection instead of the remote socket server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_remote_server(not args.local, REMOTE_SERVER_URL)
    sock = open_rover_socket()
    try:
        rover_json = fetch_rover_json(sock)
        raw_telemetry = rover_json.get("pr_telemetry")
        if not isinstance(raw_telemetry, dict):
            raise RuntimeError("ROVER.json did not contain pr_telemetry")

        telemetry = make_sanitized_telemetry(raw_telemetry)
        x_cm, y_cm, z_cm, heading_deg = parse_pose(telemetry)
        lidar = telemetry.get("lidar")
        if not isinstance(lidar, list):
            raise RuntimeError("ROVER.json did not contain a lidar array")

        print("Rover position")
        print(json.dumps(
            {
                "x_cm": round(float(x_cm), 3),
                "y_cm": round(float(y_cm), 3),
                "z_cm": round(float(z_cm), 3),
                "heading_deg": round(float(heading_deg), 3),
                "raw_world_x_m": round(float(raw_telemetry.get("rover_pos_x", 0.0)), 3),
                "raw_world_y_m": round(float(raw_telemetry.get("rover_pos_y", 0.0)), 3),
                "raw_world_z_m": round(float(raw_telemetry.get("rover_pos_z", 0.0)), 3),
            },
            indent=2,
        ))
        print()
        print("Lidar readings")
        for idx, value in enumerate(lidar):
            label = LIDAR_SENSOR_LABELS[idx] if idx < len(LIDAR_SENSOR_LABELS) else f"sensor_{idx}"
            print(f"{idx:02d} {label}: {float(value):.2f} cm")
    finally:
        close_rover_socket(sock)


if __name__ == "__main__":
    main()
