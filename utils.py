"""Small shared utilities."""

from pathlib import Path
import random
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_weights_only(path):
    """Load a raw tensor-only PyTorch state dictionary."""
    path = Path(path)
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")

    if not (
        isinstance(state_dict, dict)
        and len(state_dict) > 0
        and all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state_dict.items()
        )
    ):
        raise RuntimeError(f"{path.name} is not a raw tensor-only state dictionary.")
    return state_dict
