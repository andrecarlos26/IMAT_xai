import numpy as np
import tensorflow as tf
import random
import os
import uuid

def set_seed(seed: int | None) -> None:
    """
    If seed is not None, fixes the randomness of numpy, tensorflow and random.
    """
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

import os
import uuid

def save_summary_txt(info: list[str]) -> str:
    """
    Receives a list of strings, creates (if necessary) the 'txt' folder in the project root,
    joins the strings with line breaks, saves them in a file '{uuid}.txt' inside 'txt'
    and returns the content of the generated string.
    """
    summary = "\n".join(info)
    base_dir = os.getcwd()
    txt_dir = os.path.join(base_dir, 'txt')
    os.makedirs(txt_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}.txt"
    file_path = os.path.join(txt_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    return summary