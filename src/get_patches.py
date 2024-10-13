import os
import cv2
import glob
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def pad_image(image: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size
    return cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def crop_patch(image: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    return image[y:y + patch_size, x:x + patch_size]

def find_patch_indices(image_shape: Tuple[int, int], patch_size: int, patch_overlap: int) -> List[Tuple[int, int]]:
    height, width = image_shape[:2]
    return [(x, y) for y in range(0, height - patch_size + 1, patch_size - patch_overlap)
            for x in range(0, width - patch_size + 1, patch_size - patch_overlap)]

def process_image(image_path: Path, patch_size: int, patch_overlap: int, target_dir: Path) -> None:
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        padded_image = pad_image(image, patch_size)
        indices = find_patch_indices(padded_image.shape, patch_size, patch_overlap)
        for i, (x, y) in enumerate(indices):
            patch_path = os.path.join(target_dir, f"{os.path.basename(image_path)}_patch_{i}.png")
            if os.path.exists(patch_path):
                continue
            patch = crop_patch(padded_image, x, y, patch_size)
            cv2.imwrite(str(patch_path), patch)

        logging.info(f"Processed {image_path.name}: generated {len(indices)} patches")
    except Exception as e:
        logging.error(f"Error processing {image_path.name}: {str(e)}")

def setup_logging(log_level):
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def process_images(input_dir: str, output_dir: str, patch_size: int, patch_overlap: int, file_extension: str = '*.png'):
    path_images = glob.glob(os.path.join(input_dir, '**', file_extension), recursive=True)
    logging.info(f'Found {len(path_images)} images')

    os.makedirs(output_dir, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, Path(img_path), patch_size, patch_overlap, Path(output_dir))
                   for img_path in path_images]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing Images"):
            pass

def main(args):
    setup_logging(args.log_level)
    process_images(args.input_dir, args.output_dir, args.patch_size, args.patch_overlap, args.file_extension)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images into overlapping patches.")
    parser.add_argument('input_dir', type=str, help="Input directory containing images")
    parser.add_argument('output_dir', type=str, help="Output directory for patches")
    parser.add_argument('--patch_size', type=int, default=256, help="Size of each patch")
    parser.add_argument('--patch_overlap', type=int, default=64, help="Overlap between patches")
    parser.add_argument('--file_extension', type=str, default='*.png', help="File extension to process")
    parser.add_argument('--log_level', type=str, default='INFO', help="Logging level")
    args = parser.parse_args()

    main(args)