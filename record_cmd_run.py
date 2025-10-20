#!/usr/bin/env python3
"""Run CMD 1 for a short burst, then CMD 0, while logging BLE laser data."""

from __future__ import annotations

import asyncio
import sys
import builtins
from datetime import datetime
from typing import Optional

from bleak import BleakClient
from bleak.exc import BleakError

from bluetooth_connection import (
    DATA_WRITER,
    build_notification_handler,
    clear_buffers,
    get_laser_samples,
    parse_delimiter,
)


def _timestamped_print(*args, **kwargs) -> None:
    """Prefix output with a timestamp for easier log correlation."""
    file = kwargs.pop("file", sys.stdout)
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    flush = kwargs.pop("flush", False)
    prefix = datetime.now().strftime("[%H:%M:%S.%f] ")
    builtins.print(prefix, end="", file=file, flush=flush)
    builtins.print(*args, sep=sep, end=end, file=file, flush=flush)


print = _timestamped_print

DEVICE_ADDRESS = "60E2ECE4-761B-6B31-FD1F-6FD559C4FE52"
NOTIFY_CHAR = "0000ffe1-0000-1000-8000-00805f9b34fb"
WRITE_CHAR = "0000ffe1-0000-1000-8000-00805f9b34fb"
CMD_RUN = b"CMD-SET 40 22\n"
CMD_STOP = b"CMD 0\n"
CMD_FORWARD_30 = b"CMD-SET 22 22\n"
RUN_DURATION_SEC = 2
CONNECT_TIMEOUT = 8.0
DELIMITER = "\n"
DECODE_ERRORS = "replace"
SHOW_RAW = False
POST_STOP_DELAY = 1.0


async def run_sequence() -> None:
    clear_buffers()
    DATA_WRITER.reset()

    delimiter_bytes = parse_delimiter(DELIMITER)
    handler = build_notification_handler(delimiter_bytes, DECODE_ERRORS, SHOW_RAW)

    try:
        async with BleakClient(DEVICE_ADDRESS, timeout=CONNECT_TIMEOUT) as client:
            print("[BLE] Connected to device, starting notifications...")
            await client.start_notify(NOTIFY_CHAR, handler)
            try:
                # First, send a short forward command (25 25) for 2 seconds as requested,
                # then proceed with the original recorded sequence.
                print("[CMD] Sending prep forward 25 25 (2s)")
                await client.write_gatt_char(WRITE_CHAR, CMD_FORWARD_30, response=True)
                await asyncio.sleep(2.5)

                print("[CMD] Sending CMD 1 (start)")
                await client.write_gatt_char(WRITE_CHAR, CMD_RUN, response=True)
                await asyncio.sleep(RUN_DURATION_SEC)
                print("[CMD] Sending CMD 0 (stop)")
                await client.write_gatt_char(WRITE_CHAR, CMD_STOP, response=True)
                await asyncio.sleep(POST_STOP_DELAY)
            finally:
                try:
                    await client.stop_notify(NOTIFY_CHAR)
                except BleakError as exc:
                    print(f"[WARN] Failed to stop notifications cleanly: {exc}")
    except BleakError as exc:
        print(f"[ERROR] BLE interaction failed: {exc}")
        return

    samples = get_laser_samples()
    print(f"[BLE] Collected {len(samples)} laser samples during the run.")


def main() -> None:
    asyncio.run(run_sequence())


if __name__ == "__main__":
    main()
