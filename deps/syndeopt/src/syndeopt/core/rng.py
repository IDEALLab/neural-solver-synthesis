import random
from typing import Optional

def make_rng(seed: Optional[int] = None) -> random.Random:
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    return rng
