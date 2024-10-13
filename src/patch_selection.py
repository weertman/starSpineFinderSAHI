from get_patches import process_images
import random
import os
import glob
import shutil
import logging
from typing import List, Tuple
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_patches(output_dir: str) -> List[str]:
    return glob.glob(os.path.join(output_dir, '*.png'))


def select_random_patches(patches: List[str], n: int) -> List[str]:
    if len(patches) < n:
        logging.warning(f"Only {len(patches)} patches available. Selecting all.")
        return patches
    return random.sample(patches, n)


def create_directory(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def copy_file(src: str, dest: str) -> None:
    shutil.copyfile(src, dest)


def organize_patch(patch: str, output_dir: str, samples_per_dir: int, index: int) -> None:
    subdir = os.path.join(output_dir, str(index // samples_per_dir))
    create_directory(subdir)
    dest = os.path.join(subdir, os.path.basename(patch))
    copy_file(patch, dest)


def organize_patches_parallel(patches: List[str], output_dir: str, samples_per_dir: int) -> None:
    create_directory(output_dir)
    with ProcessPoolExecutor() as executor:
        organize_func = partial(organize_patch, output_dir=output_dir, samples_per_dir=samples_per_dir)
        list(tqdm(executor.map(organize_func, patches, range(len(patches))), total=len(patches),
                  desc="Organizing patches"))


def clean_up(dir_to_remove: str) -> None:
    shutil.rmtree(dir_to_remove)
    logging.info(f"Deleted temporary directory: {dir_to_remove}")


def process_and_organize_patches(
        input_dir: str,
        temp_dir: str,
        output_dir: str,
        patch_size: int,
        patch_overlap: int,
        file_extension: str,
        num_patches: int,
        samples_per_dir: int
) -> None:
    try:
        process_images(input_dir, temp_dir, patch_size, patch_overlap, file_extension)

        patches = get_patches(temp_dir)
        logging.info(f"Generated {len(patches)} patches")

        selected_patches = select_random_patches(patches, num_patches)

        organize_patches_parallel(selected_patches, output_dir, samples_per_dir)
        logging.info(f"Selected and organized {len(selected_patches)} patches")

        clean_up(temp_dir)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    setup_logging()

    input_dir = r'/Users/wlweert/Documents/python/starDashboard/archive'
    temp_dir = r'../patches/archive'
    output_dir = r'../patches/random_archive'

    process_and_organize_patches(
        input_dir=input_dir,
        temp_dir=temp_dir,
        output_dir=output_dir,
        patch_size=256,
        patch_overlap=64,
        file_extension='*.png',
        num_patches=1000,
        samples_per_dir=50
    )