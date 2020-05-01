import numpy as np
import pandas as pd
from nistats.design_matrix import make_design_matrix
from nistats.hemodynamic_models import _sample_condition,  _orthogonalize, _resample_regressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from pybest.utils import _load_gifti, tqdm_out
import pybest.noise_model as NM
from nilearn import masking
import os

"""
TODO:
* Fix paths
* Make sure arguments are the correct data types
"""

"""
This is dumb, I know 
But I tried to manufactor Kendricks HRFs but no banana,
so lets load em
"""
HRF_KERNELS2 = pd.read_csv('pybest/data/hrf_ts.tsv', sep='\t').values[:, 1:]
add = 10
# for k in range(20):
# 	tmp = np.hstack((np.zeros(add), HRF_KERNELS[:, k]))[:-add]
# 	HRF_KERNELS = np.c_[HRF_KERNELS, tmp]

fav = HRF_KERNELS2[:, 17]
HRF_KERNELS = np.zeros((501, 20))
HRF_KERNELS[:, 9] = fav
for i in range(1, 20):
	add = 4*i
	HRF_KERNELS[:, i] = np.hstack((np.zeros(add), fav))[:-add]

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
	"""This function returns the design matrix 
	
	Arguments:
		HRF {[array-like]} -- The HRF kernel
		frame_times {array-like} -- the time points for each volume
		trial_types {array-like} -- list of length of events, condition identifier values for each onset
		onsets {array-like} -- list of length of events, onset time in seconds of each event
		stim_durs {[type]} -- list of length of events, length of each event
		modulations {[type]} -- list of length of events, modulation of each event
	
	Keyword Arguments:
		oversampling {int} -- oversampling factor to perform the convolution, default 20
		min_onset {int} -- minimal onset relative to frame_times[0] (in seconds)
		events that start before frame_times[0] + min_onset are not considered, default 0
	
	Returns:
		[array-like] -- Design matrix which has been convolved with the HRF kernel provided (and with each column orthogonlized)
	"""
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


def get_best_HRF(func_data, run_idx, onsets, trial_types, TR, stim_durs=None, modulations=None, mask=False, work_dir=''):
	"""We fit each available HRF on each voxel in a cross-validated fashion. We return the HRF indices 
	that corresponds to the best fitting HRF for each voxel
	
	Arguments:
		func_data {[array]} -- Functional data SAMPLES x VOXELS
		run_idx {[array]} -- list of run indices
		onsets {array-like} -- list of length of events, onset time in seconds of each event
		trial_types {array-like} -- list of length of events, condition identifier values for each onset
		TR {float} -- Repetition time for functional data
	
	Keyword Arguments:
		stim_durs {[type]} -- list of length of events, length of each event
		modulations {[type]} -- list of length of events, modulation of each event
	
	Returns:
		[array] -- the indices for the best fitting HRF for each voxel
	"""

	# how many HRFs do we ahve
	n_HRFs = HRF_KERNELS.shape[1]

	n_run = np.unique(run_idx).size

	# Go to default settings with 1 second durations
	# and 1s as scaling factors for all events
	if np.all(stim_durs == None):
		stim_durs = np.ones(len(onsets))
	if np.all(modulations == None):
		modulations = np.ones(len(onsets))

	# Hard code this for now
	min_onset = 0.0
	# our HRF kernels are sampled at 10 hz
	oversampling = 10 * TR

	# Check so we have at least 2 runs?
	assert np.unique(run_idx).size > 1, "You need at least 2 runs"

	# get specifics
	n_vols, n_voxels = func_data.shape

	# timing of samples
	frame_times = np.arange(n_vols) * TR

	# pre-allocate HRF x VOXEL
	r2s_hrf = np.zeros((n_HRFs, n_voxels))

	# loop over HRFs
	for i in tqdm(range(n_HRFs), desc=f'Calculating R2 for HRF:', file=tqdm_out):
		HRF_kernel = HRF_KERNELS[: ,i]

		# get regressor matrix for this HRF kernel
		X = get_regressor_matrix(HRF_kernel, frame_times, trial_types, onsets,
								stim_durs, modulations, oversampling=oversampling, min_onset=-25)
		X = np.c_[X, np.ones(X.shape[0])]
		# fit data
		betas = np.linalg.inv(X.T @ X) @ X.T @ func_data
		y_pred = X @ betas
		r2s_hrf[i, :] = r2_score(func_data, y_pred, multioutput='raw_values')


		# loop over runs, fit each run by itself to save memory
		# r2_scores = np.zeros(func_data.shape[1])
		# for run in np.unique(run_idx):
		# 	run_mask = run_idx == run
		# 	y = func_data[run_mask, :]
		# 	run_X = X[run_mask, :]
		# 	# our design matrix have alot columns that might not
		# 	# be occuring during this run - lets remove them
		# 	run_X = run_X[:, ~np.all(run_X == 0, axis=0)]

		# 	# fit data
		# 	betas = np.linalg.inv(run_X.T @ run_X) @ run_X.T @ y

		# 	# Overfitting to check
		# 	y_pred = run_X @ betas
		# 	r2_scores += r2_score(y, y_pred, multioutput='raw_values')

		# r2s_hrf[i, :] = r2_scores / n_run
	
	# pick best HRF for each voxel
	best_HRF = r2s_hrf.argmax(0)

	if mask:
		r2_img = masking.unmask(r2s_hrf, mask)
		f_out = os.path.join(work_dir, 'HRF_r2.nii.gz')
		r2_img.to_filename(f_out)

	return best_HRF


def optimize_signal_model(func_data, run_idx, onsets, trial_types, TR, stim_durs=None, modulations=None):
	"""Fits the data with the preferred HRF of each voxel
	
	Arguments:
		func_data {[array]} -- Functional data SAMPLES x VOXELS
		run_idx {[array]} -- list of run indices
		onsets {array-like} -- list of length of events, onset time in seconds of each event
		trial_types {array-like} -- list of length of events, condition identifier values for each onset
		TR {float} -- Repetition time for functional data
	
	Keyword Arguments:
		stim_durs {[type]} -- list of length of events, length of each event
		modulations {[type]} -- list of length of events, modulation of each event
	
	Returns:
		fitted_brain {array} -- The data fit using preferred HRFs for each voxel, of size func_data.shape
		r2 {array} -- The R-square for each voxel
	"""
	ಠ_ಠ = ValueError('You need at least 2 runs')
	assert np.unique(run_idx).size > 1, ಠ_ಠ

	# Go to default settings with 1 second durations
	# and 1s as scaling factors for all events
	if np.all(stim_durs == None):
		stim_durs = np.ones(len(onsets))
	if np.all(modulations == None):
		modulations = np.ones(len(onsets))

	# get indices of the best HRF per voxel
	best_HRF = get_best_HRF(func_data, run_idx, onsets,
							trial_types, TR, stim_durs=stim_durs, modulations=modulations)
	
	# get specifics
	n_vols, n_voxels = func_data.shape
	frame_times = np.arange(n_vols) * TR

	# fit voxels with individual HRFs
	# So I could only think of one good way of doing this
	# - Lets fit all voxels with the same HRF together instead of one voxel alone
	HRF_idx = np.unique(best_HRF)
	fitted_brain = np.zeros(func_data.shape)
	
	# over unique HRFs
	for HRF in HRF_idx:
	
		# this is the HRF we are using
		HRF_kernel = HRF_KERNELS[:, HRF]

		# make mask so we know which voxels have this HRF
		mask = best_HRF == HRF

		# get regressor matrix for this HRF kernel
		X = get_regressor_matrix(HRF_kernel, frame_times, trial_types, onsets,
								stim_durs, modulations, oversampling=oversampling, min_onset=-25)
		X = np.c_[X, np.ones(X.shape[0])]

		# fit data
		betas = np.linalg.inv(X.T @ X) @ X.T @ func_data[:, mask]

		fitted_brain[:, mask] = X @ betas

	r2 = r2_score(func_data, fitted_brain, multioutput='raw_values')

	return fitted_brain, r2


if __name__ == "__main__":
	"""
	Putting a unit test here for now
	"""
	# for consistency
	np.random.seed(1)

	# create data
	n_vox = 400
	func_data = np.random.random((900, n_vox))*2
	# demean data
	func_data -= func_data.mean(0)
	# create run indicator, just one run for now
	run_idx = np.repeat([0, 1, 2], 300)
	# define a few onsets
	onsets = np.array([4, 30, 240, 340, 410, 440, 550, 620, 700, 750, 810])
	# its a single trial, so each onset is a new trial type
	trial_types = np.arange(len(onsets))
	# sluggish TR, just what we need
	TR = 1
	
	# test our function that only returns the regressors for now
	time = np.arange(func_data.shape[0]) * TR

	# lets add signal based on one of the HRFs
	stim_durs = np.ones(len(onsets))
	modulations = np.ones(len(onsets))
	exp_condition = (onsets,
					stim_durs,
					modulations)
	frame_times = np.arange(func_data.shape[0]) * TR

	# lets pick an HRF for each voxel
	hrfs = np.random.choice(np.arange(20), n_vox)

	for vox in range(n_vox):
		HRF = HRF_KERNELS[:, hrfs[vox]]

		# get the regressor for this condition
		reg = compute_regressor(
			exp_condition, HRF, frame_times,
			oversampling=10 * TR,
			min_onset=0)

		func_data[:, vox] += reg.flatten()

	best_HRF = get_best_HRF(func_data, run_idx, onsets, trial_types,
							TR, stim_durs=stim_durs, modulations=modulations)
	print("We guessed", np.mean(best_HRF==hrfs)*100, "% correct")
	assert np.all(best_HRF==hrfs), "The chosen HRFs for the voxels isn't correct"

	# This is what the function call looks like
	# maybe its best to send in a 
	fitted_brain, r2 = optimize_signal_model(func_data, run_idx, onsets,
							trial_types, TR, stim_durs=stim_durs, modulations=modulations)

	# unit_test 2
	from nilearn import plotting

	sub = '02'
	ses = '1'
	task = 'face'
	work_dir = 'fsaverage6/work'
	func_data, event_data, mask, run_idx = NM.load_denoised_data(sub, ses, task, work_dir)

	TR = 0.7

	frame_times = np.arange(func_data.shape[0]) * TR
	
	event = event_data.copy()

	# the event timings are based on the start of each run
	# we need them to be in relation to the first run only
	runs = np.unique(event.run)
	runs.sort()
	full_onset = event[event.run==runs[0]]['onset'].values
	for r in runs[1:]:
		max_prev = max(full_onset)
		full_onset = np.hstack((full_onset, event[event.run==r]['onset'].values+max_prev))
	
	event['onset'] = full_onset
	event['trial_type'] = np.arange(len(event)) # single trial


	"""
	Lets fit as we usually do
	"""
	import warnings
	warnings.simplefilter("ignore")
	X = make_design_matrix(
			frame_times, event, hrf_model='glover', drift_order=2, drift_model=None)

	X = X.values
	betas = np.linalg.inv(X.T @ X) @ X.T @ func_data
	y_pred = X @ betas
	r2 = r2_score(func_data, y_pred, multioutput='raw_values')
	print(f'Range of R2: {np.nanmin(r2):.5f} - {np.nanmax(r2):.5f}')

	"""
	Lets fit our functions
	"""

	onsets = event['onset'].values
	trial_types = event['trial_type'].values
	stim_durs = event['duration'].values

	# best HRF
	best_HRF = get_best_HRF(func_data, run_idx, onsets, trial_types, TR)
	a = [sum(x==best_HRF) for x in range(40)]
	print(a)
	plt.plot(a)
	plt.show()

	# FULL FIT
	fitted_brain, r2 = optimize_signal_model(func_data, run_idx, onsets,
							trial_types, TR)
	print(f'Range of R2: {np.nanmin(r2):.5f} - {np.nanmax(r2):.5f}')

	
	r2_img = masking.unmask(r2, mask)

	coords = range(-30, 50, 5)

	title = 'R2'
	display = plotting.plot_stat_map(
		r2_img, colorbar=False, cut_coords=coords,
		display_mode='z', draw_cross=False,
		title=title, black_bg=False, annotate=False)
