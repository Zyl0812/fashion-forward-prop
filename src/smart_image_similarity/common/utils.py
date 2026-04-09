import os
import random
import re

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set process-wide random seeds for reproducible training."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sorted_alphanum(values):
    convert = lambda value: int(value) if value.isdigit() else value.lower()
    alphanum = lambda value: [convert(x) for x in re.split(r"([0-9]+)", value)]
    return sorted(values, key=alphanum)
