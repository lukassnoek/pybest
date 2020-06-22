import os.path as op
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


HRF_MODELS = [
    'kay', 'spm', 'spm + derivative', 'spm + derivative + dispersion',
    'glover', 'glover + derivative', 'glover + derivative + dispersion', 'fir'
]

here = op.dirname(__file__)
HRFS = pd.read_csv(op.join(here, 'data', 'hrf_ts.tsv'), sep='\t', index_col=0)
# Note: Kendrick's HRFs are defined at 0.1 sec resolution
t_hrf = HRFS.index.copy()

# Resample to msec resolution
hrf_oversampling = 10
t_high = np.linspace(0, 50, num=HRFS.shape[0] * hrf_oversampling, endpoint=True)
HRFS_HR = np.zeros((t_high.size, 20))
for i in range(20):  # should be able to do this w/o for loop, but lazy
    f = interp1d(t_hrf, HRFS.iloc[:, i].to_numpy())
    HRFS_HR[:, i] = f(t_high)  # hr = high resolution


STATS = dict(beta='effect_size', zscore='z_score')
