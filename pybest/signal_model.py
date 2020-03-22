import numpy as np
import pandas as pd
from nistats.design_matrix import make_design_matrix
from nistats.hemodynamic_models import _sample_condition,  _orthogonalize, _resample_regressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

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

	* THIS FUNCTION IS REAPPRORIATED FROM:
	https://github.com/nistats/nistats/blob/a2a39284ed7dc004e1c288809a65104b7cf3aa12/nistats/hemodynamic_models.py#L235
	nistats hemodynamic_models

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

def get_regressor_matrix(HRF, frame_times, trial_types, onsets, stim_durs, modulations, oversampling=20, min_onset=0):
	# loop over conditions
	regressor_matrix = None
	for condition in np.unique(trial_types):
		condition_mask = (trial_types == condition)
		# create condition array
		exp_condition = (onsets[condition_mask],
							stim_durs[condition_mask],
							modulations[condition_mask])
		# get the regressor for this condition
		reg = compute_regressor(
			exp_condition, HRF, frame_times,
			oversampling=oversampling,
			min_onset=min_onset)
		
		# append regressor to full regressor matrix
		if regressor_matrix is None:
			regressor_matrix = reg
		else:
			regressor_matrix = np.hstack((regressor_matrix, reg))
	return regressor_matrix


def optimize_signal_model(data, run_indicator, onsets, trial_types, TR, stim_durs=None, modulations=None):

	# TODO better solution than loading the tsv?
	hrf_ts = pd.read_csv('pybest/data/hrf_ts.tsv', sep='\t').values[:, 1:]
	n_HRFs = hrf_ts.shape[1]

	# Go to default settings with 1 second durations
	# and 1s as scaling factors for all events
	if stim_durs == None:
		stim_durs = np.ones(len(onsets))
	if modulations == None:
		modulations = np.ones(len(onsets))

	# Hard code this for now
	min_onset = 0.0
	# our HRF kernels are sampled at 10 hz
	oversampling = 10 * TR

	# TODO check so we have at least 2 runs?
	runs = np.unique(run_indicator)

	# loop over runs
	r = runs[0]
	r_rest = runs[]

	# get specific run data 
	run_data = data[run_indicator==r, :]
	n_vols, n_voxels = run_data.shape

	# timing of samples
	frame_times = np.arange(n_vols) * TR
	# pre-allocate HRF x VOXEL
	r2s_hrf = np.zeros((n_HRFs, n_voxels))

	# loop over HRFs
	for i in range(n_HRFs):
		HRF_kernel = hrf_ts[: ,i]
		# get regressor matrix for this HRF kernel
		X = get_regressor_matrix(HRF_kernel, frame_times, trial_types, onsets,
								stim_durs, modulations, oversampling=oversampling, min_onset=0)

		betas = np.linalg.inv(X.T @ X) @ (X.T @ run_data)
		fit = X @ betas 

		r2s_hrf[i, :] = r2_score(run_data, fit, multioutput='raw_values')

	# pick best HRF for each voxel
	best_HRF = r2s_hrf.argmax(0)

	return best_HRF


# create data
data = np.random.random((900, 4))
# demean data
data = data - data.mean(0)
# create run indicator, just one run for now
run_indicator = np.zeros(len(data))
# define a few onsets
onsets = np.array([4, 44, 100, 122, 166, 188, 242, 402])
# its a single trial, so each onset is a new trial type
trial_types = np.arange(len(onsets))
# sluggish TR, just what we need
TR = 1
# test our function that only returns the regressors for now
time = np.arange(data.shape[0]) * TR

# lets add signal based on one of the HRFs
# TODO see if we recovered the HRF choice we made
stim_durs = np.ones(len(onsets))
modulations = np.ones(len(onsets))
exp_condition = (onsets,
				stim_durs,
				modulations)
frame_times = np.arange(data.shape[0]) * TR
hrf_ts = pd.read_csv('pybest/data/hrf_ts.tsv', sep='\t').values[:, 1:]
HRF = hrf_ts[:, 3]

# get the regressor for this condition
reg = compute_regressor(
	exp_condition, HRF, frame_times,
	oversampling=10/TR,
	min_onset=0)

data = data + reg/1.2
plt.plot(time, data[:, 0])
plt.plot(time, reg/1.2)
plt.show()