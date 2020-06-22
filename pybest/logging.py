import logging
from tqdm import tqdm
from datetime import datetime
from functools import partial


def get_logger(verbose):
    logging.basicConfig(
        level=getattr(logging, verbose),
        format="%(asctime)s [%(levelname)-7.7s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('pybest')
    return logger

# Custom tqdm progress bar (to play nicely with the logger)
tqdm_ctm = partial(tqdm, bar_format='{desc}  {bar}  {n_fmt}/{total_fmt}')


def tdesc(s):
    # Custom text for `desc` parameter of tqdm
    return datetime.now().strftime('%Y-%m-%d %H:%M [INFO   ]') + f'  {s}'
