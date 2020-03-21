import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-8s] [%(levelname)-7.7s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('pybest')

class Dataset:
    pass

def preprocess():
    pass