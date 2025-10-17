#!/usr/bin/env python3
"""BLE connectivity probe using the bleak library."""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, List, Optional

try:  # pragma: no cover - optional dependency guard
    from bleak import BleakClient, BleakScanner, BleakError  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - environment check
    print("[ERROR] bleak is not installed. Install it with 'pip install bleak'", file=sys.stderr)
    raise SystemExit(1) from exc


@dataclass
class LaserSample:
    angle: int
    distance: int


@dataclass
class MpuStatus:
    degree_x: int
    count_run1: int
    count_run2: int


class ParsedDataPool:
    def __init__(self, max_laser_samples: int = 500):
        self._laser_samples: Deque[LaserSample] = deque(maxlen=max_laser_samples)
        self._mpu_status: Optional[MpuStatus] = None
        self._lock = threading.Lock()

    def add_laser_sample(self, sample: LaserSample) -> None:
        with self._lock:
            self._laser_samples.append(sample)

    def set_mpu_status(self, status: MpuStatus) -> None:
        with self._lock:
            self._mpu_status = status

    def get_laser_samples(self, limit: Optional[int] = None) -> List[LaserSample]:
        with self._lock:
            if limit is None or limit >= len(self._laser_samples):
                return list(self._laser_samples)
            else:
                return list(self._laser_samples)[-limit:]

    def get_mpu_status(self) -> Optional[MpuStatus]:
        with self._lock:
            return self._mpu_status

    def clear(self) -> None:
        with self._lock:
            self._laser_samples.clear()
            self._mpu_status = None


DATA_POOL = ParsedDataPool(max_laser_samples=1000)


class ParsedDataWriter:
    def __init__(self, parsed_path: Path, raw_path: Path) -> None:
        self.parsed_path = parsed_path
        self.raw_path = raw_path
        self._lock = threading.Lock()
        self._init_files()

    def _init_files(self) -> None:
        for path in (self.parsed_path, self.raw_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8"):
                pass

    def write_laser_point(self, sample: LaserSample, raw_bytes: bytes, idx: int) -> None:
        parsed_line = f"LASER idx={idx} angle={sample.angle} distance={sample.distance}"
        raw_line = "LASER_RAW " + " ".join(f"0x{b:02X}" for b in raw_bytes)
        self._append(parsed_line, raw_line)

    def write_mpu(self, status: MpuStatus, raw_bytes: bytes) -> None:
        parsed_line = (
            f"MPU degree_x={status.degree_x} count1={status.count_run1} count2={status.count_run2}"
        )
        raw_line = "MPU_RAW " + " ".join(f"0x{b:02X}" for b in raw_bytes)
        self._append(parsed_line, raw_line)

    def _append(self, parsed_line: str, raw_line: str) -> None:
        with self._lock:
            with self.parsed_path.open("a", encoding="utf-8") as parsed_file:
                parsed_file.write(parsed_line + "\n")
            with self.raw_path.open("a", encoding="utf-8") as raw_file:
                raw_file.write(raw_line + "\n")


DEFAULT_PARSED_FILE = Path("ble_parsed.log")
DEFAULT_RAW_FILE = Path("ble_raw.log")
DATA_WRITER = ParsedDataWriter(DEFAULT_PARSED_FILE, DEFAULT_RAW_FILE)


def get_laser_samples(limit: Optional[int] = None) -> List[LaserSample]:
    """Return recent laser samples (optionally limited to the last *limit* items)."""
    return DATA_POOL.get_laser_samples(limit)


def get_mpu_status() -> Optional[MpuStatus]:
    """Return the most recently parsed MPU status frame, if any."""
    return DATA_POOL.get_mpu_status()


def clear_buffers() -> None:
    """Reset all cached frames (mainly for tests)."""
    DATA_POOL.clear()


def parse_payload(message: str, binary_mode: bool) -> bytes:
    if binary_mode:
        if len(message) % 2 != 0:
            raise ValueError("Hex payload must contain an even number of characters")
        return bytes.fromhex(message)
    return message.encode("ascii", errors="replace")


def parse_delimiter(value: str) -> bytes:
    """Interpret escape sequences ("\r", "\n", "\r\n") or hex literals like 0x0d0a."""
    if value.startswith("0x") and len(value) > 2:
        try:
            return bytes.fromhex(value[2:])
        except ValueError as err:  # pragma: no cover - CLI validation
            raise ValueError(f"Invalid hex delimiter '{value}': {err}")
    try:
        decoded = value.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError as err:  # pragma: no cover - CLI validation
        raise ValueError(f"Invalid escape sequence in delimiter '{value}': {err}") from err
    return decoded.encode("ascii", errors="ignore")


async def scan_devices(timeout: float) -> None:
    print(f"[INFO] Scanning for BLE devices for {timeout:.1f}s...")
    try:
        devices = await BleakScanner.discover(timeout=timeout)
    except BleakError as err:
        print(f"[ERROR] BLE scan failed: {err}")
        return
    if not devices:
        print("[WARN] No BLE devices discovered.")
        return
    for device in devices:
        print(f"Device found: {device.name} ({device.address})")



def _bytes_to_hex(payload: bytes) -> str:
    return " ".join(f"{byte:02X}" for byte in payload) if payload else ""


class FrameParser:
    HEADER_LASER = 0xA5
    HEADER_MPU = 0xA6
    LASER_PAIR_COUNT = 250
    FRAME_LEN_LASER = 5  # head + angle + distance
    FRAME_LEN_MPU = 11   # head + deg + cnt1 + cnt2

    def __init__(
        self,
        data_pool: ParsedDataPool,
        writer: ParsedDataWriter,
        debug_hex: bool = False,
    ) -> None:
        self._buffer = bytearray()
        self._pool = data_pool
        self._debug_hex = debug_hex
        self._laser_count = 0
        self._writer = writer

    def feed(self, chunk: bytes) -> None:
        if self._debug_hex:
            print(f"[DATA] chunk(hex)={_bytes_to_hex(chunk)}")
        self._buffer.extend(chunk)
        self._drain()

    def _drain(self) -> None:
        while True:
            if not self._buffer:
                return
            header = self._buffer[0]
            if header == self.HEADER_LASER:
                frame_len = self.FRAME_LEN_LASER
            elif header == self.HEADER_MPU:
                frame_len = self.FRAME_LEN_MPU
            else:
                # 头部不匹配，丢弃一个字节尝试重新同步
                lost = self._buffer.pop(0)
                print(f"[WARN] Dropping unknown header byte 0x{lost:02X}")
                continue

            if len(self._buffer) < frame_len:
                return  # 等待更多数据

            frame = bytes(self._buffer[:frame_len])
            del self._buffer[:frame_len]
            try:
                if header == self.HEADER_LASER:
                    self._handle_laser_frame(frame)
                else:
                    self._handle_mpu_frame(frame)
            except Exception as exc:
                print(f"[ERROR] Failed to parse frame {frame.hex()}: {exc}")

    def _handle_laser_frame(self, frame: bytes) -> None:
        angle = int.from_bytes(frame[1:3], byteorder="little", signed=False)
        distance = int.from_bytes(frame[3:5], byteorder="little", signed=False)
        sample = LaserSample(angle=angle, distance=distance)
        self._pool.add_laser_sample(sample)
        group_idx = self._laser_count % self.LASER_PAIR_COUNT
        self._laser_count += 1
        self._writer.write_laser_point(sample, bytes(frame), group_idx)
        print(f"[LASER] idx={group_idx} angle={angle} distance={distance}")

    def _handle_mpu_frame(self, frame: bytes) -> None:
        degree_x = int.from_bytes(frame[1:3], byteorder="little", signed=True)
        count_run1 = int.from_bytes(frame[3:7], byteorder="little", signed=False)
        count_run2 = int.from_bytes(frame[7:11], byteorder="little", signed=False)
        status = MpuStatus(degree_x=degree_x, count_run1=count_run1, count_run2=count_run2)
        self._pool.set_mpu_status(status)
        print(f"[MPU] degree_x={degree_x} count1={count_run1} count2={count_run2}")
        self._writer.write_mpu(status, bytes(frame))


def build_notification_handler(
    delimiter: bytes,
    _: str,
    print_raw_chunks: bool,
) -> Callable[[object, bytearray], None]:
    parser = FrameParser(DATA_POOL, DATA_WRITER, debug_hex=print_raw_chunks)

    def handler(_: object, data: bytearray) -> None:
        chunk = bytes(data)
        if delimiter:
            parser.feed(chunk)
        else:
            parser.feed(chunk)

    return handler


async def connect_and_probe(args: argparse.Namespace, payload: bytes, delimiter: bytes) -> None:
    print(f"[INFO] Connecting to {args.address} (timeout={args.connect_timeout:.1f}s)...")
    try:
        async with BleakClient(
            args.address,
            timeout=args.connect_timeout,
            adapter=args.adapter,
        ) as client:
            print("[OK] Connected to device.")

            if args.list_services:
                print("[INFO] Available services/characteristics:")
                for service in client.services:
                    print(f"  Service {service.uuid}")
                    for char in service.characteristics:
                        props = ",".join(char.properties)
                        print(f"    Char {char.uuid}  props=[{props}]")

            if args.notify_char:
                print(f"[INFO] Subscribing to notifications on {args.notify_char}...")
                handler = build_notification_handler(delimiter, args.decode_errors, args.show_raw)
                await client.start_notify(args.notify_char, handler)

            if payload and args.write_char:
                print(f"[INFO] Writing {len(payload)} bytes to {args.write_char} (response={'no' if args.write_without_response else 'yes'})")
                await client.write_gatt_char(
                    args.write_char,
                    payload,
                    response=not args.write_without_response,
                )
            elif payload and not args.write_char:
                print("[WARN] Payload provided but --write-char missing; skipping write.")

            if args.read_char:
                try:
                    data = await client.read_gatt_char(args.read_char)
                except BleakError as err:
                    print(f"[ERROR] Failed to read {args.read_char}: {err}")
                else:
                    try:
                        text = data.decode("ascii", errors=args.decode_errors)
                    except UnicodeDecodeError:
                        text = None
                    if text is not None:
                        print(f"[OK] Read {len(data)} bytes (ASCII): {text}")
                    else:
                        print(f"[OK] Read {len(data)} bytes: {data!r}")

            if args.notify_char:
                wait_desc = "Ctrl+C to stop" if args.duration <= 0 else f"waiting {args.duration:.1f}s"
                print(f"[INFO] Listening for notifications ({wait_desc})...")
                try:
                    if args.duration <= 0:
                        while True:
                            await asyncio.sleep(1.0)
                    else:
                        await asyncio.sleep(args.duration)
                except KeyboardInterrupt:
                    print("\n[INFO] Notification stream interrupted by user.")
                finally:
                    try:
                        await client.stop_notify(args.notify_char)
                    except BleakError:
                        pass
            else:
                if args.duration > 0:
                    await asyncio.sleep(args.duration)

    except BleakError as err:
        print(f"[ERROR] BLE operation failed: {err}")
        raise SystemExit(2) from err


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe a BLE robot using bleak.")
    parser.add_argument("address", nargs="?", help="BLE device address (e.g. AA:BB:CC:DD:EE:FF)")
    parser.add_argument("--adapter", help="Bluetooth adapter identifier (if multiple adapters are present)")
    parser.add_argument("--connect-timeout", type=float, default=8.0, help="BLE connection timeout in seconds (default: 8.0)")

    parser.add_argument("--write-char", help="Characteristic UUID used for writing commands")
    parser.add_argument("--notify-char", help="Characteristic UUID used for notifications")
    parser.add_argument("--read-char", help="Characteristic UUID to read once after connecting")
    parser.add_argument("--write-without-response", action="store_true", help="Use write-without-response when sending payload")

    parser.add_argument("--message", default="ping", help="ASCII message to send to the write characteristic")
    parser.add_argument("--binary", action="store_true", help="Interpret --message as hex bytes (e.g. 'ff00aa')")
    parser.add_argument("--no-write", action="store_true", help="Skip sending the probe message even if provided")

    parser.add_argument("--stream", action="store_true", help="Deprecated alias: notifications are streamed automatically when --notify-char is set")
    parser.add_argument(
        "--delimiter",
        default="\n",
        help="Delimiter for decoding notification lines (default: newline). Accepts escape sequences or hex prefixed with 0x",
    )
    parser.add_argument("--decode-errors", choices=["strict", "ignore", "replace"], default="replace", help="Error handling for ASCII decoding (default: replace)")
    parser.add_argument("--show-raw", action="store_true", help="Print raw chunks alongside decoded lines")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to keep the connection open (<=0 means until Ctrl+C)")

    parser.add_argument("--scan", action="store_true", help="Scan for nearby BLE devices and exit")
    parser.add_argument("--scan-timeout", type=float, default=6.0, help="Scan duration in seconds (default: 6.0)")
    parser.add_argument("--list-services", action="store_true", help="List services and characteristics after connecting")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.scan:
        asyncio.run(scan_devices(args.scan_timeout))
        if not args.address:
            return

    if not args.address:
        parser.error("address is required unless only scanning")

    try:
        payload = parse_payload(args.message, args.binary)
    except ValueError as err:
        parser.error(str(err))

    try:
        delimiter = parse_delimiter(args.delimiter)
    except ValueError as err:
        parser.error(str(err))

    if args.no_write:
        payload = b""

    asyncio.run(connect_and_probe(args, payload, delimiter))


if __name__ == "__main__":
    main()
