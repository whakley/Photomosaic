#!/usr/bin/env python3
"""
Photomosaic Generator
======================
Recreates a target image as a mosaic built from a folder of "tile" images.
Each tile in the output grid is chosen because its average color closely
matches the corresponding region of the target image.

Requirements:
    pip install pillow numpy

Basic usage:
    python photomosaic.py target.jpg tiles_folder/ -o output.jpg

More control:
    python photomosaic.py target.jpg tiles_folder/ -o output.jpg \
        --tile-size 24 --resize 1.5 --blend 0.2 --spread-usage
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

# Optional HEIC/HEIF support (iPhone photos). Only enabled if pillow-heif
# is installed: pip install pillow-heif
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    SUPPORTED_EXTS.update({".heic", ".heif"})
except ImportError:
    pass


def fit_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize + center-crop an image to exactly (target_w, target_h)
    without distorting its aspect ratio."""
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _folder_signature(folder: Path, paths, tile_w: int, tile_h: int) -> str:
    """A signature that changes if the tile size changes, or if any file
    in the folder is added, removed, renamed, or modified."""
    parts = [f"tile={tile_w}x{tile_h}"]
    for p in paths:
        st = p.stat()
        parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode("utf-8", "ignore")).hexdigest()


def _cache_path(folder: Path, tile_w: int, tile_h: int) -> Path:
    return folder / f".photomosaic_cache_{tile_w}x{tile_h}.npz"


def load_tiles(tiles_folder: str, tile_w: int, tile_h: int, use_cache: bool = True):
    """Load every supported image in tiles_folder, resize/crop each to
    (tile_w, tile_h), and compute its average RGB color.

    Results are cached to disk (one cache file per tile size, stored inside
    the tiles folder). On the next run, if no files were added/removed/
    modified, the cache is reused instead of re-reading every image.
    """
    folder = Path(tiles_folder)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)

    if not paths:
        raise ValueError(f"No supported images found in '{tiles_folder}'")

    cache_file = _cache_path(folder, tile_w, tile_h)
    signature = _folder_signature(folder, paths, tile_w, tile_h)

    if use_cache and cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                if str(data["signature"]) == signature:
                    arrays = data["tile_arrays"]  # (N, H, W, 3) uint8
                    colors = data["colors"]  # (N, 3) float64
                    tiles = [Image.fromarray(arr) for arr in arrays]
                    print(f"Using cached tiles ({len(tiles)} tiles, no re-read needed).")
                    return tiles, colors
                else:
                    print("Tiles folder changed since last cache — rebuilding...")
        except Exception as e:
            print(f"Cache unreadable ({e}); rebuilding...")

    tiles, colors = [], []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"  skipping {p.name}: {e}")
            continue

        img = fit_crop(img, tile_w, tile_h)
        arr = np.asarray(img, dtype=np.float64)
        colors.append(arr.reshape(-1, 3).mean(axis=0))
        tiles.append(img)

    if not tiles:
        raise ValueError("No tile images could be loaded.")

    colors = np.array(colors)

    if use_cache:
        try:
            tile_arrays = np.stack([np.asarray(t, dtype=np.uint8) for t in tiles])
            np.savez_compressed(
                cache_file,
                signature=signature,
                tile_arrays=tile_arrays,
                colors=colors,
            )
            print(f"Cached {len(tiles)} tiles to '{cache_file.name}' for instant reuse next time.")
        except Exception as e:
            print(f"Could not write cache ({e}); continuing without it.")

    return tiles, colors


def build_mosaic(
    target_path: str,
    tiles_folder: str,
    tile_size: int = 32,
    resize_factor: float = 1.0,
    blend: float = 0.0,
    spread_usage: bool = False,
    repeat_penalty: float = 4000.0,
    use_cache: bool = True,
) -> Image.Image:
    target = Image.open(target_path).convert("RGB")

    if resize_factor != 1.0:
        new_size = (
            max(1, round(target.width * resize_factor)),
            max(1, round(target.height * resize_factor)),
        )
        target = target.resize(new_size, Image.LANCZOS)

    tile_w = tile_h = tile_size
    cols = target.width // tile_w
    rows = target.height // tile_h
    if cols == 0 or rows == 0:
        raise ValueError("tile_size is larger than the (resized) target image.")

    # Crop so the target divides evenly into the grid
    target = target.crop((0, 0, cols * tile_w, rows * tile_h))
    target_arr = np.asarray(target, dtype=np.float64)

    print(f"Loading tiles from '{tiles_folder}' ...")
    tiles, tile_colors = load_tiles(tiles_folder, tile_w, tile_h, use_cache=use_cache)
    print(f"Loaded {len(tiles)} tiles. Building a {cols}x{rows} grid ({cols * rows} cells)...")

    if spread_usage and len(tiles) < cols * rows:
        print(
            f"Note: only {len(tiles)} unique tiles for {cols * rows} cells — "
            "some repeats are unavoidable even with --spread-usage."
        )

    mosaic = Image.new("RGB", target.size)
    used_counts = np.zeros(len(tiles), dtype=np.float64)
    progress_step = max(1, rows // 10)

    for r in range(rows):
        for c in range(cols):
            cell = target_arr[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w]
            avg_color = cell.reshape(-1, 3).mean(axis=0)

            dists = np.sum((tile_colors - avg_color) ** 2, axis=1)
            if spread_usage:
                dists = dists + used_counts * repeat_penalty
            idx = int(np.argmin(dists))
            used_counts[idx] += 1

            tile_img = tiles[idx]
            if blend > 0:
                tile_arr = np.asarray(tile_img, dtype=np.float64)
                blended = tile_arr * (1 - blend) + avg_color * blend
                tile_img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

            mosaic.paste(tile_img, (c * tile_w, r * tile_h))

        if (r + 1) % progress_step == 0 or r == rows - 1:
            print(f"  row {r + 1}/{rows} done")

    return mosaic


def main():
    parser = argparse.ArgumentParser(
        description="Create a photomosaic from a target image and a folder of tile images."
    )
    parser.add_argument("target", help="Path to the target image (the picture to recreate)")
    parser.add_argument("tiles_folder", help="Folder containing the tile images")
    parser.add_argument("-o", "--output", default="mosaic_output.jpg", help="Output file path")
    parser.add_argument(
        "--tile-size", type=int, default=32,
        help="Width/height of each tile in pixels (default: 32). Smaller = more detail, slower."
    )
    parser.add_argument(
        "--resize", type=float, default=1.0,
        help="Scale factor applied to the target before tiling (default: 1.0). "
             "Increase for a higher-resolution, more detailed mosaic."
    )
    parser.add_argument(
        "--blend", type=float, default=0.0,
        help="0.0-1.0: how much to tint each tile toward the target's color in that cell "
             "(default: 0.0 = no tint, pure tile images). Try 0.15-0.3 for closer resemblance."
    )
    parser.add_argument(
        "--spread-usage", action="store_true",
        help="Discourage reusing the same tile too often, for more visual variety."
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Don't use/write the tile cache — always re-read every image from disk."
    )
    args = parser.parse_args()

    mosaic = build_mosaic(
        target_path=args.target,
        tiles_folder=args.tiles_folder,
        tile_size=args.tile_size,
        resize_factor=args.resize,
        blend=args.blend,
        spread_usage=args.spread_usage,
        use_cache=not args.no_cache,
    )

    mosaic.save(args.output, quality=95)
    print(f"Saved mosaic to '{args.output}' ({mosaic.width}x{mosaic.height})")


if __name__ == "__main__":
    main()