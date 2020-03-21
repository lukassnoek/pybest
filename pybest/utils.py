import logging
import numpy as np
import nibabel as nib


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-8s] [%(levelname)-7.7s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('pybest')


def _load_gifti(f):
    """ Load gifti array. """
    f_gif = nib.load(f)
    return np.vstack([arr.data for arr in f_gif.darrays])
