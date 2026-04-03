"""CLI tool: connect to ESP32-Radar over BT SPP and print LD2450 targets.

Usage:
    python scripts/radar_bt_receiver.py --port /dev/rfcomm0

Pair the ESP32 first:
    bluetoothctl
    > scan on          # find ESP32-Radar MAC
    > pair <MAC>
    > trust <MAC>
    > quit
    sudo rfcomm bind 0 <MAC>   # creates /dev/rfcomm0
"""
from __future__ import annotations

import argparse
import time

from drone_rl.utils.radar import RadarReceiver


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream LD2450 radar data from ESP32 over Bluetooth SPP"
    )
    parser.add_argument(
        "--port", default="/dev/rfcomm0",
        help="Serial port for BT SPP connection (default: /dev/rfcomm0)"
    )
    parser.add_argument("--baud", type=int, default=9600)
    args = parser.parse_args()

    r = RadarReceiver(port=args.port, baud=args.baud)
    r.start()
    print(f"Connected to {args.port} at {args.baud} baud. Press Ctrl+C to stop.\n")
    try:
        while True:
            obs = r.latest
            t1_x, t1_y, t1_s = obs[0], obs[1], obs[2]
            t2_x, t2_y, t2_s = obs[3], obs[4], obs[5]
            t3_x, t3_y, t3_s = obs[6], obs[7], obs[8]
            print(
                f"T1: x={t1_x:7.0f}mm  y={t1_y:7.0f}mm  spd={t1_s:5.0f}cm/s  |  "
                f"T2: x={t2_x:7.0f}mm  y={t2_y:7.0f}mm  spd={t2_s:5.0f}cm/s  |  "
                f"T3: x={t3_x:7.0f}mm  y={t3_y:7.0f}mm  spd={t3_s:5.0f}cm/s"
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        r.stop()


if __name__ == "__main__":
    main()
