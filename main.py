from typing import Callable, List, Tuple
import os
import torch
import logging
import catalyst
from catalyst import utils

is_fp16_used = False
logging.basicConfig(filename="app.log", filemode="w", format="%(name)s %(level)s %(message)s", level=logging.INFO)
logging.info(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

logging.info("Start")

logging.info("Data preparation")
from pathlib import Path
ROOT = Path("segmentation_data/")
train_image_path = ROOT / "train"
train_mask_path = ROOT / "train_masks"
test_image_path = ROOT / "test"

ALL_IMAGES = sorted(train_image_path.glob("*.jpg"))
logging.info(f"Num images: {len(ALL_IMAGES)}")

ALL_MASKS = sorted(train_mask_path.glob("*.gif"))
logging.info(f"Num maks: {len(ALL_MASKS)}")

import random
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread as gif_imread
from catalyst import utils

def show_examples(name: str, image: np.ndarray, mask: np.ndarray, figname: str):
    plt.figure(figsize=(10, 14))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title(f"Mask: {name}")

    plt.savefig(figname)
    plt.close()

def show(index: int, images: List[Path], masks: List[Path], figname: str, transforms=None)-> None:
    image_path = images[index]
    name = image_path.name

    image = utils.imread(image_path)
    mask = gif_imread(masks[index])

    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp["image"]
        mask = temp["mask"]

    show_examples(name, image, mask, figname)

def show_random(images: List[Path], masks: List[Path], figname: str, transforms=None) -> None:
    length = len(images)
    index = random.radint(0, length - 1)
    show(index, images, masks, figname, transforms)

show_random(ALL_IMAGES, ALL_MASKS, "figure1.png")

logging.info("Done!")