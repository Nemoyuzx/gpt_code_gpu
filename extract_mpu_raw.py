#!/usr/bin/env python3
"""Extract MPU_RAW entries from a BLE raw log file."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MPU_RAW lines from ble_raw logs.")
    parser.add_argument(
        "input",
        nargs="?",
        default="ble_raw.log",
        help="Path to the ble_raw log file (default: ble_raw.log)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to write the extracted MPU_RAW lines; prints to stdout if omitted.",
    )
    return parser.parse_args()


def extract_lines(input_path: Path, output_path: Path | None) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input log not found: {input_path}")

    if output_path:
        with input_path.open("r", encoding="utf-8") as src, output_path.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                if line.startswith("MPU_RAW"):
                    dst.write(line)
    else:
        with input_path.open("r", encoding="utf-8") as src:
            for line in src:
                if line.startswith("MPU_RAW"):
                    print(line.rstrip())


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser() if args.output else None
    extract_lines(input_path, output_path)


if __name__ == "__main__":
    main()
