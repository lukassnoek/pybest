import io
import logging
import numpy as np
import nibabel as nib
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-8s] [%(levelname)-7.7s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('pybest')


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    
    def flush(self):
        self.logger.log(self.level, self.buf)

tqdm_out = TqdmToLogger(logger, level=logging.INFO)


def _load_gifti(f):
    """ Load gifti array. """
    f_gif = nib.load(f)
    return np.vstack([arr.data for arr in f_gif.darrays])
