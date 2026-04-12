from __future__ import annotations

import argparse
import time

from rover_control import (
    close_rover_socket,
    configure_remote_server,
    fetch_ltv_json,
    open_rover_socket,
    send_float_command,
)


USE_REMOTE_SERVER = False
REMOTE_SERVER_URL = "http://35.3.249.68:5001"
CMD_LTV_PING = 2050
CMD_LTV_PING_UNLIMITED = 2051
USE_UNLIMITED_PING = False
PING_RESPONSE_TIMEOUT_SEC = 1.5
PING_RESPONSE_POLL_SEC = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one LTV ping and print the signal change.")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use the remote socket server instead of the default UDP TSS connection.",
    )
    parser.add_argument(
        "--url",
        default=REMOTE_SERVER_URL,
        help="Remote socket server URL to use with --remote.",
    )
    parser.add_argument(
        "--unlimited",
        action="store_true",
        help="Use the unlimited/debug ping command (2051) instead of the normal ping (2050).",
    )
    return parser.parse_args()


def read_ltv_signal_strength(sock) -> float:
    ltv_json = fetch_ltv_json(sock)
    signal = ltv_json.get("signal", {})
    return float(signal.get("strength", 0.0))


def request_ping_and_read_strength(sock, *, use_unlimited_ping: bool) -> tuple[float, float]:
    ping_command = CMD_LTV_PING_UNLIMITED if use_unlimited_ping else CMD_LTV_PING
    previous_strength = read_ltv_signal_strength(sock)
    send_float_command(sock, ping_command, 1.0)
    deadline = time.monotonic() + PING_RESPONSE_TIMEOUT_SEC
    latest_strength = previous_strength
    while time.monotonic() < deadline:
        time.sleep(PING_RESPONSE_POLL_SEC)
        latest_strength = read_ltv_signal_strength(sock)
        if latest_strength != previous_strength:
            return (previous_strength, latest_strength)
    return (previous_strength, latest_strength)


def main() -> None:
    args = parse_args()
    configure_remote_server(args.remote, args.url)
    sock = open_rover_socket()
    try:
        before_strength, after_strength = request_ping_and_read_strength(
            sock,
            use_unlimited_ping=bool(args.unlimited or USE_UNLIMITED_PING),
        )
        print(f"remote={bool(args.remote)}")
        if args.remote:
            print(f"url={args.url}")
        print(f"ping_command={CMD_LTV_PING_UNLIMITED if (args.unlimited or USE_UNLIMITED_PING) else CMD_LTV_PING}")
        print(f"signal_before={before_strength:.3f}")
        print(f"signal_after={after_strength:.3f}")
        print(f"signal_changed={after_strength != before_strength}")
    finally:
        close_rover_socket(sock)


if __name__ == "__main__":
    main()
