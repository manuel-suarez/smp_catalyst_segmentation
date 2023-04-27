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

logging.info("Done!")