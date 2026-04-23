#!/usr/bin/env python3
"""Extract parsed MPU entries from a BLE parsed log file."""

import argparse
import csv
import sys
from pathlib import Path
import re
from typing import Iterable, TextIO
from output_paths import BLE_PARSED_LOG, ensure_parent

MPU_LINE_RE = re.compile(
    r"MPU\s+degree_x=(?P<degree>-?\d+)\s+count1=(?P<count1>-?\d+)\s+count2=(?P<count2>-?\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract parsed MPU records (degree_x/count1/count2) from ble_parsed logs."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(BLE_PARSED_LOG),
        help=f"Path to the ble_parsed log file (default: {BLE_PARSED_LOG})",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to write the extracted MPU entries; prints to stdout if omitted.",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("text", "csv"),
        default="text",
        help="Output format: original text lines or CSV rows (default: text).",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not write CSV header when using --format=csv.",
    )
    return parser.parse_args()


def iter_mpu_records(lines: Iterable[str]):
    for line in lines:
        match = MPU_LINE_RE.search(line)
        if match:
            yield (
                int(match.group("degree")),
                int(match.group("count1")),
                int(match.group("count2")),
                line.rstrip(),
            )


def write_text(records: Iterable[tuple[int, int, int, str]], out: TextIO) -> None:
    for *_values, raw_line in records:
        out.write(raw_line + "\n")


def write_csv(records: Iterable[tuple[int, int, int, str]], out: TextIO, *, header: bool) -> None:
    writer = csv.writer(out)
    if header:
        writer.writerow(["degree_x", "count1", "count2"])
    for degree, count1, count2, _ in records:
        writer.writerow([degree, count1, count2])


def extract_parsed(input_path: Path, output_path: Path | None, *, fmt: str, header: bool) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input log not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as src:
        entries = list(iter_mpu_records(src))

    if not entries:
        return

    if output_path:
        output_path = ensure_parent(output_path)
        with output_path.open("w", encoding="utf-8", newline="") as dst:
            _write(entries, dst, fmt, header)
    else:
        _write(entries, sys.stdout, fmt, header)


def _write(entries, dst: TextIO, fmt: str, header: bool) -> None:
    if fmt == "csv":
        write_csv(entries, dst, header=header)
    else:
        write_text(entries, dst)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser() if args.output else None
    header = False if args.no_header else True
    extract_parsed(input_path, output_path, fmt=args.format, header=header)


if __name__ == "__main__":
    main()
