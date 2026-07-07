from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray:
    """Read an image as a BGR numpy array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def list_image_files(folder: Path) -> list[Path]:
    """List supported image files in a tile folder."""
    if not folder.exists():
        raise FileNotFoundError(f"Tile folder does not exist: {folder}")
    files = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise ValueError(f"No image files found in: {folder}")
    return files


def crop_to_block_multiple(image: np.ndarray, block_size: int) -> np.ndarray:
    """Crop image so width and height are divisible by block_size."""
    height, width = image.shape[:2]
    new_height = (height // block_size) * block_size
    new_width = (width // block_size) * block_size
    return image[:new_height, :new_width].copy()


def mean_lab_feature(image_bgr: np.ndarray) -> np.ndarray:
    """Return the Lab mean-color feature [mean_L, mean_a, mean_b]."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)


def load_tile_library(tile_folder: Path, block_size: int, seed: int | None = None) -> tuple[list[np.ndarray], list[str]]:
    """Load tile images and assign tile IDs by list order.

    The tile ID is simply the position in the list:
    tiles[0] -> index 0, tiles[1] -> index 1, ...

    If seed is provided, the file order is shuffled. This makes the tile
    ordering a recovery condition.
    """
    files = list_image_files(tile_folder)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(files)

    tiles: list[np.ndarray] = []
    names: list[str] = []
    for file_path in files:
        tile = read_image(file_path)
        tile = cv2.resize(tile, (block_size, block_size), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
        names.append(file_path.name)
    return tiles, names


def build_tile_features(tiles: list[np.ndarray]) -> np.ndarray:
    """Compute Lab mean-color features for all tiles."""
    return np.vstack([mean_lab_feature(tile) for tile in tiles])


def encode_secret_to_indices(
    secret_image: np.ndarray,
    tiles: list[np.ndarray],
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a secret image into a 2D tile-index map."""
    secret_image = crop_to_block_multiple(secret_image, block_size)
    height, width = secret_image.shape[:2]
    rows = height // block_size
    cols = width // block_size

    tile_features = build_tile_features(tiles)
    index_map = np.zeros((rows, cols), dtype=np.int32)

    for row in range(rows):
        for col in range(cols):
            y0 = row * block_size
            y1 = y0 + block_size
            x0 = col * block_size
            x1 = x0 + block_size
            block = secret_image[y0:y1, x0:x1]

            block_feature = mean_lab_feature(block)
            distances = np.linalg.norm(tile_features - block_feature, axis=1)
            index_map[row, col] = int(np.argmin(distances))

    return secret_image, index_map


def render_mosaic(index_map: np.ndarray, tiles: list[np.ndarray], block_size: int) -> np.ndarray:
    """Render a mosaic image from a tile-index map."""
    rows, cols = index_map.shape
    mosaic = np.zeros((rows * block_size, cols * block_size, 3), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            tile_id = int(index_map[row, col]) % len(tiles)
            y0 = row * block_size
            y1 = y0 + block_size
            x0 = col * block_size
            x1 = x0 + block_size
            mosaic[y0:y1, x0:x1] = tiles[tile_id]

    return mosaic


def save_index_json(
    output_path: Path,
    index_map: np.ndarray,
    tile_names: list[str],
    block_size: int,
    tile_seed: int | None,
) -> None:
    """Save the tile-index sequence and metadata for embedding."""
    payload = {
        "role": "intermediate_index_sequence",
        "scheme": "MosaicIndexRepresentation_MeanLab",
        "description": "Secret image is converted to tile index code by Lab mean-color matching.",
        "block_size_pixels": block_size,
        "rows": int(index_map.shape[0]),
        "cols": int(index_map.shape[1]),
        "symbol_count": int(index_map.size),
        "tile_seed": tile_seed,
        "tile_count": len(tile_names),
        "tile_names_in_order": tile_names,
        "index_map_flat": index_map.reshape(-1).astype(int).tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def create_tile_atlas(tiles: list[np.ndarray], block_size: int, output_path: Path) -> None:
    """Create a visual atlas of the tile library order."""
    cols = int(math.ceil(math.sqrt(len(tiles))))
    rows = int(math.ceil(len(tiles) / cols))
    atlas = np.zeros((rows * block_size, cols * block_size, 3), dtype=np.uint8)

    for tile_id, tile in enumerate(tiles):
        row, col = divmod(tile_id, cols)
        y0 = row * block_size
        y1 = y0 + block_size
        x0 = col * block_size
        x1 = x0 + block_size
        atlas[y0:y1, x0:x1] = tile

    cv2.imwrite(str(output_path), atlas)


def convert(secret_path: Path, tile_folder: Path, output_dir: Path, block_size: int, tile_seed: int | None) -> None:
    """Run the mosaic-index conversion pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    secret = read_image(secret_path)
    tiles, tile_names = load_tile_library(tile_folder, block_size, seed=tile_seed)
    cropped_secret, index_map = encode_secret_to_indices(secret, tiles, block_size)
    mosaic = render_mosaic(index_map, tiles, block_size)

    stem = secret_path.stem
    cropped_secret_path = output_dir / f"{stem}_cropped.png"
    mosaic_path = output_dir / f"{stem}_mosaic_b{block_size}.png"
    index_path = output_dir / f"{stem}_mosaic_index_b{block_size}.json"
    atlas_path = output_dir / f"tile_atlas_b{block_size}.png"

    cv2.imwrite(str(cropped_secret_path), cropped_secret)
    cv2.imwrite(str(mosaic_path), mosaic)
    save_index_json(index_path, index_map, tile_names, block_size, tile_seed)
    create_tile_atlas(tiles, block_size, atlas_path)

    print("Done.")
    print(f"Cropped secret image: {cropped_secret_path}")
    print(f"Mosaic image: {mosaic_path}")
    print(f"Index code JSON: {index_path}")
    print(f"Tile atlas: {atlas_path}")
    print(f"Block size: {block_size}x{block_size} pixels")
    print(f"Index count: {index_map.size}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a secret image into a mosaic tile-index representation.")
    parser.add_argument("--secret", required=True, help="Path to the secret image.")
    parser.add_argument("--tiles", required=True, help="Folder containing tile images.")
    parser.add_argument("--out", default="outputs/mosaic", help="Output folder.")
    parser.add_argument("--block-size", type=int, default=16, help="Mosaic block size in pixels.")
    parser.add_argument("--tile-seed", type=int, default=15, help="Seed for shuffling tile order. Use -1 to keep file order.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tile_seed = None if args.tile_seed == -1 else args.tile_seed
    convert(
        secret_path=Path(args.secret),
        tile_folder=Path(args.tiles),
        output_dir=Path(args.out),
        block_size=args.block_size,
        tile_seed=tile_seed,
    )


if __name__ == "__main__":
    main()

