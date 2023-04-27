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

from typing import List
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(
            self,
            images: List[Path],
            masks: List[Path] = None,
            transforms = None
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = utils.imread(image_path)

        result = {"image": image}

        if self.masks is not None:
            mask = gif_imread(self.masks[idx])
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(**result)

        result["filename"] = image_path.name

        return result

import albumentations as albu
from albumentations.pytorch import ToTensor

def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]

def hard_transforms():
    result = [
        albu.RandomRotate90(),
        albu.Cutout(),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3),
        albu.HueSaturationValue(p=0.3)
    ]
    return result

def resize_transforms(image_size=224):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
        albu.SmallestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )
    ])
    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])
    random_crop_big = albu.Compose([
        albu.LongestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )
    ])
    # Converts the image to a square of size image_size x image_size
    result = [
        albu.OneOf([
            random_crop,
            rescale,
            random_crop_big
        ], p=1)
    ]
    return result

def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensor()]

def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result

train_transforms = compose([
    resize_transforms(),
    hard_transforms(),
    post_transforms()
])
valid_transforms = compose([pre_transforms(), post_transforms()])
show_transforms = compose([resize_transforms(), hard_transforms()])

show_random(ALL_IMAGES, ALL_MASKS, "figure2.png", transforms=show_transforms)

logging.info("Loaders")
import collections
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def get_loaders(
        images: List[Path],
        masks: List[Path],
        random_state: int,
        valid_size: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transforms_fn = None,
        valid_transforms_fn = None,
) -> dict:
    indices = np.arange(len(images))

    # Let's divide the data set into train and valid parts.
    train_indices, valid_indices = train_test_split(
        indices, test_size=valid_size, random_state=random_state, shuffle=True
    )
    np_images = np.array(images)
    np_masks = np.array(masks)
    # Creates our train dataset
    train_dataset = SegmentationDataset(
        images=np_images[train_indices].tolist(),
        masks=np_masks[train_indices].tolist(),
        transforms=train_transforms_fn
    )
    # Creates our valid dataset
    valid_dataset = SegmentationDataset(
        images=np_images[valid_indices].tolist(),
        masks=np_masks[valid_indices].tolist(),
        transforms=valid_transforms_fn
    )
    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )
    # And expect to get an OrderedDict of Loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader
    return loaders
if is_fp16_used:
    batch_size = 64
else:
    batch_size = 32
logging.info(f"batch_size: {batch_size}")
loaders = get_loaders(
    images=ALL_IMAGES,
    masks=ALL_MASKS,
    random_state=SEED,
    train_transforms_fn=train_transforms,
    valid_transforms_fn=valid_transforms,
    batch_size=batch_size
)
loaders.info("Model definition")
import segmentation_models_pytorch as smp
# We will use Feature Pyramid Network with pre-trained ResNeXt50 backbone
model = smp.FPN(encoder_name="resnext50_32x4d", classes=1)
loaders.info("Model training")
from torch import nn
from catalyst.contrib.nn import DiceLoss, IoULoss
# we have multiple criterions
criterion = {
    "dice": DiceLoss(),
    "iou": IoULoss(),
    "bce": nn.BCEWithLogitsLoss()
}
from torch import optim
from catalyst.contrib.nn import RAdam, Lookahead
learning_rate = 0.001
encoder_learning_rate = 0.0005
# Since we use a pre-trained encoder, we will reduce the learning rate on it.
layerwise_params = {"encoder": dict(lr=encoder_learning_rate, weight_decay=0.00003)}
# This function removes weight_decay for biases and applies our layerwise_params
model_params = utils.process_model_params(model, layerwise_params=layerwise_params)
# Catalyst has new SOTA optimizers out of box
base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
optimizer = Lookahead(base_optimizer)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

from catalyst.dl import SupervisedRunner
num_epochs = 3
logdir = "./logs/segmentation"
device = utils.get_device()
logging.info(f"Device: {device}")

if is_fp16_used:
    fp16_params = dict(opt_level="01")
else:
    fp16_params = None
logging.info(f"FP16 params: {fp16_params}")

# by default SupervisedRunner uses "features" and "targets",
# in our case we get "image" and "mask" keys in dataset __getitem__
runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")

logging.info("Training")
from catalyst.dl import DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback
from catalyst.contrib.callbacks import DrawMasksCallback

callbacks = [
    # Each criterion is calculated separately.
    CriterionCallback(
        input_key="mask",
        prefix="loss_dice",
        criterion_key="dice"
    ),
    CriterionCallback(
        input_key="mask",
        prefix="loss_iou",
        criterion_key="iou"
    ),
    CriterionCallback(
        input_key="mask",
        prefix="loss_bce",
        criterion_key="bce"
    ),
    # And only then we aggregate everything into one loss.
    MetricAggregationCallback(
        prefix="loss",
        mode="weighted_sum", # can be "sum", "weighted_sum", or "mean"
        # because we want weighted sum, we need to add scale for each loss
        metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
    ),
    # metrics
    DiceCallback(input_key="mask"),
    IouCallback(input_key="mask"),
    # visualization
    DrawMasksCallback(output_key='logits',
                      input_image_key='image',
                      input_mask_key='mask',
                      summary_step=50
                      )
]
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    # our dataloaders
    loaders=loaders,
    # we can specify the callbacks list for the experiment:
    callbacks=callbacks,
    # path to save logs
    logdir=logdir,
    num_epochs=num_epochs,
    # save our best checkpoint by IoU metric
    main_metric="iou",
    # IoU needs to be maximized.
    minimize_metric=False,
    # for FP16.
    fp16=fp16_params,
    # print train logs
    verbose=True
)

# Model inference
TEST_IMAGES = sorted(test_image_path.glob("*.jpg"))
# create test dataset
test_dataset = SegmentationDataset(
    TEST_IMAGES,
    transforms=valid_transforms
)
num_workers: int = 4
infer_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)
# this get predictions for the whole loader
predictions = np.vstack(list(map(
    lambda x: x["logits"].cpu().numpy(),
    runner.predict_loader(loader=infer_loader, resume=f"{logdir}/checkpoints/best.pth")
)))
logging.info(type(predictions))
logging.info(predictions.shape)

threshold = 0.5
max_count = 5

for i, (features, logits) in enumerate(zip(test_dataset, predictions)):
    image = utils.tensor_to_ndimage(features["image"])

    mask_ = torch.from_numpy(logits[0]).sigmoid()
    mask = utils.detach(mask_ > threshold).astype("float")

    show_examples(name="", image=image, mask=mask)

    if i >= max_count:
        break

logging.info("Done!")