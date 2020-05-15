import numpy as np

HRF_MODELS = [
    'kay', 'spm', 'spm + derivative', 'spm + derivative + dispersion',
    'glover', 'glover + derivative', 'glover + derivative + dispersion', 'fir'
]
ALPHAS = np.array([0, 0.01, 1, 10, 100, 500, 1000, 5000])
