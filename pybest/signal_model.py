import numpy as np
import pandas as pd
from nistats.design_matrix import make_design_matrix
from nistats.hemodynamic_models import _sample_condition,  _orthogonalize, _resample_regressor

"""
Input: 	2D (time x voxels) denoised signal array
		run indicator array
		TR
		stim_dur - duration of stimuli (list)
		onset - onset times
		trial_type - if single trial its just np.arange(len(onset))

Step 1: for run in runs:
	for hrf_model in hrf_models:
		Create LSA design, fit, compute R2; keep track of R2
Step 2: compute median (mean?) R2 curve across HRF-models for each voxel and pick max for each voxel!
"""


def compute_regressor(exp_condition, hrf_kernel, frame_times,
					  oversampling=10, min_onset=0):
	""" This is the main function to convolve regressors with hrf model
	Parameters
	----------
	exp_condition : array-like of shape (3, n_events)
		yields description of events for this condition as a
		(onsets, durations, amplitudes) triplet
	hrf_kernel : array-like of shape (501)
			one of the kernels from Kendricks library (10 hz sample rate, 501 samples)
		
	frame_times : array of shape (n_scans)
		the desired sampling times
	oversampling : int, optional
		oversampling factor to perform the convolution
	min_onset : float, optional
		minimal onset relative to frame_times[0] (in seconds)
		events that start before frame_times[0] + min_onset are not considered
	Returns
	-------
	computed_regressors: array of shape(n_scans, 1)
		computed regressors sampled at frame times
	Notes
	-----
	"""
	oversampling = int(oversampling)

	# this is the average tr in this session, not necessarily the true tr
	tr = float(frame_times.max()) / (np.size(frame_times) - 1)

	# 1. create the high temporal resolution regressor
	hr_regressor, hr_frame_times = _sample_condition(
		exp_condition, frame_times, oversampling, min_onset)

	# 2. convolve the regressor and hrf, and downsample the regressor
	conv_reg = np.array([np.convolve(hr_regressor, hrf_kernel)[:hr_regressor.size]])

	# 3. temporally resample the regressors
	computed_regressors = _resample_regressor(
		conv_reg, hr_frame_times, frame_times)

	# 4. ortogonalize the regressors
	computed_regressors = _orthogonalize(computed_regressors)
	
	return computed_regressors


from scipy.io import loadmat
import matplotlib.pyplot as plt
hrf_ts = loadmat('pybest/data/hrf_ts.mat')['hrf_ts']


hrf_ts = pd.read_csv('pybest/data/hrf_ts.tsv', sep='\t').values[:, 1:]
t = np.linspace(0, 50, hrf_ts.shape[0])

plt.figure(figsize=(15, 5))
plt.plot(t,hrf_ts)
plt.xlim(0, 50)
plt.show()

# 600 samples, TR is 2 seconds
fake_data = np.random.rand(40)
n = len(fake_data)
TR = 0.5

frame_times = (np.arange(n)) * TR
oversampling = 10*TR

min_onset = 0
min_onset = float(min_onset)


# lets pick 10 random times 
onsets = np.random.choice(frame_times, 2, replace=False)
onsets = np.array([0, 5])
# 1 for scaling
modulations = np.ones(len(onsets))
# 1 second long
durations = np.ones(len(onsets))
# lets say we have 2 conditions
trial_type = np.arange(len(onsets))
np.random.shuffle(stim_type)


# loop
regressor_matrix = None

for condition in np.unique(trial_type):
	condition_mask = (trial_type == condition)
	exp_condition = (onsets[condition_mask],
						durations[condition_mask],
						modulations[condition_mask])
	reg = compute_regressor(
		exp_condition, hrf_ts[:, 10], frame_times,
		oversampling=oversampling,
		min_onset=min_onset)

	if regressor_matrix is None:
		regressor_matrix = reg
	else:
		regressor_matrix = np.hstack((regressor_matrix, reg))

plt.plot(frame_times, regressor_matrix)



def optimize_signal_model(data, run_indicator, onsets, trial_types, TR, stim_durs=None, modulations=None):

	# go to default settings with 1 seconnd durations
	# and 1s as scaling factors for all events
	if stim_durs == None:
		stim_durs = np.ones(len(onsets))
	if modulations == None:
		modulations = np.ones(len(onsets))

	# Hard code this for now
	min_onset = 0.0
	# our HRF kernels are sampled at 10 hz
	oversampling = 10 * TR

	r = 0

	# get specific run data 
	run = data[run_indicator==r, :]
	n = sum(run_indicator==r)

	# timing of samples
	frame_time = np.arange(n) * TR
	
	# loop over HRFs
	HRF = hrf_ts[: ,10]
	# get regressor matrix for this HRF kernel
	regressor_matrix = None

	for condition in np.unique(trial_type):
		condition_mask = (trial_type == condition)
		exp_condition = (onsets[condition_mask],
							durations[condition_mask],
							modulations[condition_mask])
		reg = compute_regressor(
			exp_condition, HRF, frame_times,
			oversampling=oversampling,
			min_onset=min_onset)

		if regressor_matrix is None:
			regressor_matrix = reg
		else:
			regressor_matrix = np.hstack((regressor_matrix, reg))
	


