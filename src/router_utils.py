import math
from pathlib import Path
import numpy as np
from utils.io_utils import ensure_dir
def json_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and math.isnan(x):
        return None
    return str(x)
