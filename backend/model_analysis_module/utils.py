import numpy as np
import tensorflow as tf
import random

def set_seed(seed: int | None) -> None:
    """
    If seed is not None, fixes the randomness of numpy, tensorflow and random.
    """
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
