from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def psnr(original: np.ndarray, modified: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * math.log10((255.0 * 255.0) / mse))


def symbol_to_mrt_coords(symbol: int, m: int, n: int, perm_seed: int) -> tuple[int, ...]:
    """Map one tile index symbol to N base-M coordinates."""
    total_symbols = m**n
    rng = random.Random(perm_seed)
    permutation = list(range(total_symbols))
    rng.shuffle(permutation)

    mapped_symbol = permutation[int(symbol) % total_symbols]
    coords = [0] * n
    for i in range(n - 1, -1, -1):
        coords[i] = mapped_symbol % m
        mapped_symbol //= m
    return tuple(coords)


def pixel_positions(pixel_count: int, pixel_seed: int) -> list[int]:
    """Generate a seeded embedding pixel order."""
    positions = list(range(pixel_count))
    rng = random.Random(pixel_seed)
    rng.shuffle(positions)
    return positions


def embed_symbols_to_carrier(
    carrier_bgr: np.ndarray,
    symbols: list[int],
    m: int,
    n: int,
    perm_seed: int,
    pixel_seed: int,
) -> tuple[np.ndarray, list[int]]:
    """Embed tile index symbols into the Y channel by modulo-M adjustment."""
    ycrcb = cv2.cvtColor(carrier_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.int32)

    required_pixels = len(symbols) * n
    if required_pixels > y_channel.size:
        raise ValueError(f"Carrier capacity is not enough: need {required_pixels}, got {y_channel.size}.")

    positions = pixel_positions(y_channel.size, pixel_seed)
    coords_stream: list[int] = []
    write_index = 0

    for symbol in symbols:
        coords = symbol_to_mrt_coords(symbol, m, n, perm_seed)
        for coord in coords:
            pos = positions[write_index]
            old_value = int(y_channel.flat[pos])
            target_remainder = int(coord) % m

            delta = (target_remainder - (old_value % m)) % m
            if delta > m // 2:
                delta -= m

            y_channel.flat[pos] = np.clip(old_value + delta, 0, 255)
            coords_stream.append(target_remainder)
            write_index += 1

    ycrcb[:, :, 0] = y_channel.astype(np.uint8)
    stego_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return stego_bgr, coords_stream


def embed(index_json_path: Path, carrier_path: Path, output_dir: Path, m: int, n: int, perm_seed: int, pixel_seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    index_payload = read_json(index_json_path)
    symbols = [int(value) for value in index_payload["index_map_flat"]]
    carrier = read_image(carrier_path)

    if max(symbols, default=0) >= m**n:
        raise ValueError(f"M^N is too small. M={m}, N={n}, M^N={m**n}, max symbol={max(symbols)}.")

    stego, coords_stream = embed_symbols_to_carrier(carrier, symbols, m, n, perm_seed, pixel_seed)
    stego_psnr = psnr(carrier, stego)

    stem = carrier_path.stem
    stego_path = output_dir / f"{stem}_stego_M{m}_N{n}.png"
    key_path = output_dir / f"{stem}_embedding_key_M{m}_N{n}.json"
    cv2.imwrite(str(stego_path), stego)

    write_json(
        key_path,
        {
            "role": "embedding_recovery_key",
            "scheme": "MRT-SIE_modM_Y_channel",
            "M": m,
            "N": n,
            "perm_seed": perm_seed,
            "pixel_seed": pixel_seed,
            "symbol_count": len(symbols),
            "coords_per_symbol": n,
            "required_pixels": len(symbols) * n,
            "index_rows": index_payload.get("rows"),
            "index_cols": index_payload.get("cols"),
            "block_size_pixels": index_payload.get("block_size_pixels"),
            "tile_seed": index_payload.get("tile_seed"),
            "tile_names_in_order": index_payload.get("tile_names_in_order"),
            "stego_psnr_db": stego_psnr,
            "coords_stream_flat": coords_stream,
        },
    )

    print("Done.")
    print(f"Stego image: {stego_path}")
    print(f"Embedding key JSON: {key_path}")
    print(f"M={m}, N={n}, capacity symbols={m**n}")
    print(f"Embedded symbols: {len(symbols)}")
    print(f"Required pixels: {len(symbols) * n}")
    print(f"Stego PSNR: {stego_psnr:.4f} dB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed mosaic tile index code into a carrier image with MRT-SIE.")
    parser.add_argument("--index", required=True, help="Mosaic index JSON generated by mosaic_collage_converter.py.")
    parser.add_argument("--carrier", required=True, help="Carrier image path.")
    parser.add_argument("--out", default="outputs/stego", help="Output folder.")
    parser.add_argument("--m", type=int, default=4, help="MRT base M.")
    parser.add_argument("--n", type=int, default=4, help="MRT dimension N.")
    parser.add_argument("--perm-seed", type=int, default=13579, help="Seed for symbol-to-coordinate permutation.")
    parser.add_argument("--pixel-seed", type=int, default=24680, help="Seed for embedding pixel order.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    embed(Path(args.index), Path(args.carrier), Path(args.out), args.m, args.n, args.perm_seed, args.pixel_seed)


if __name__ == "__main__":
    main()

