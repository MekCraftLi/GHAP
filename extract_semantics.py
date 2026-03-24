# extract_semantics.py
# Batch semantic mask extraction using rembg (U-2-Net).
#
# Install dependencies:
#   pip install rembg[gpu] onnxruntime-gpu pillow tqdm numpy
#
# Usage:
#   python extract_semantics.py -s ./playroom

import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
from rembg import remove, new_session
from tqdm import tqdm

# Allow loading truncated/corrupted images instead of raising an error.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Fallback mask dimensions (H x W) — must match your dataset resolution exactly.
FALLBACK_H = 832
FALLBACK_W = 1264

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def extract_masks(src_dir: Path, dst_dir: Path, session) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ])

    if not image_paths:
        print(f'[WARN] No images found in {src_dir}')
        return

    for img_path in tqdm(image_paths, desc='Extracting masks', unit='img'):
        out_path = dst_dir / (img_path.stem + '.png')
        try:
            with Image.open(img_path) as img:
                img.load()  # Force full decode to catch truncated data early.
                img_rgb = img.convert('RGB')

            result = remove(img_rgb, session=session)  # Returns RGBA PIL image.
            alpha = np.array(result)[:, :, 3]          # Extract alpha channel.
            mask = (alpha > 127).astype(np.uint8) * 255  # Binarize to 0/255.

            Image.fromarray(mask, mode='L').save(out_path)

        except Exception as e:
            print(f'\n[ERROR] Failed on {img_path.name}: {e}')
            print(f'        Saving fallback black mask -> {out_path.name}')
            fallback = np.zeros((FALLBACK_H, FALLBACK_W), dtype=np.uint8)
            Image.fromarray(fallback, mode='L').save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description='Extract binary semantic foreground masks using rembg.'
    )
    parser.add_argument(
        '-s', '--source', required=True,
        help='Dataset root directory (must contain an "images" subfolder).'
    )
    args = parser.parse_args()

    root = Path(args.source)
    src_dir = root / 'images'
    dst_dir = root / 'edge_masks'

    if not src_dir.exists():
        raise FileNotFoundError(f'Input folder not found: {src_dir}')

    print(f'Input  : {src_dir}')
    print(f'Output : {dst_dir}')
    print('Loading rembg session (u2net) ...')

    session = new_session('u2net')

    extract_masks(src_dir, dst_dir, session)

    print(f'\nDone. Masks saved to: {dst_dir}')


if __name__ == '__main__':
    main()
