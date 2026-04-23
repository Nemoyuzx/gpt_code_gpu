import argparse
import json
from pathlib import Path
from typing import Iterable, List


def _scaled_point(point: Iterable[float], factor: float) -> List[float]:
    # Round to avoid long floating-point tails while keeping precision reasonable.
    scaled = []
    for value in point:
        result = value * factor
        if abs(result - round(result)) < 1e-9:
            scaled.append(int(round(result)))
        else:
            scaled.append(round(result, 6))
    return scaled


def scale_maze(data: dict, factor: float) -> dict:
    if "segments" in data:
        for segment in data["segments"]:
            if "start" in segment:
                segment["start"] = _scaled_point(segment["start"], factor)
            if "end" in segment:
                segment["end"] = _scaled_point(segment["end"], factor)
    if "start_point" in data:
        data["start_point"] = _scaled_point(data["start_point"], factor)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale maze coordinates in a JSON file by a given factor or width ratio."
    )
    parser.add_argument("input", type=Path, help="Path to the source maze JSON file.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Path for the scaled maze JSON file (defaults to input path).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Direct scale factor to apply to all coordinates.",
    )
    parser.add_argument(
        "--original-width",
        type=float,
        help="Original maze width in meters to compute the scale factor.",
    )
    parser.add_argument(
        "--target-width",
        type=float,
        help="Target maze width in meters to compute the scale factor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.scale is not None:
        factor = args.scale
    elif args.original_width and args.target_width:
        if args.original_width == 0:
            raise ValueError("original-width must be non-zero to compute a scale factor")
        factor = args.target_width / args.original_width
    else:
        raise ValueError(
            "Specify either --scale or both --original-width and --target-width to determine the scale factor."
        )

    with args.input.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    scaled_data = scale_maze(data, factor)

    output_path = args.output if args.output else args.input
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(scaled_data, outfile, indent=1)
        outfile.write("\n")


if __name__ == "__main__":
    main()
