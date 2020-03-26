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
from sklearn.base import RegressorMixin, BaseEstimator
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

class Ridge(RegressorMixin, BaseEstimator):
	def __init__(self, lambd=1.0):
		self.lambd = lambd
	
	def fit(self, X, y, apply_to):
		"""
		Just like a normal Ridge, but with the option of only applying the penalty on certain regressors,
		that is, our noise regressors.
		
		Arguments:
			X {[array]} -- Design matrix (time x predictor)
			y {[array]} -- Data (time x voxel)
		
		Keyword Arguments:
			apply_to {bool} -- [list indicating which predictors to apply ridge to] (default: {False})
		"""
		n_predictors = X.shape[1]
		assert len(apply_to) == n_predictors, "apply_to needs to have a boolean value for each predictor"
		
		I = np.zeros((n_predictors, n_predictors))
		I[np.eye(n_predictors, dtype=bool)] = apply_to
		self.b_ = np.linalg.inv(X.T @ X + self.lambd * I) @ X.T @ y 

		return self
	
	def predict(X, y):
		return X @ self.b_ 

# load Kendricks kernels 
HRF_KERNELS = pd.read_csv('pybest/data/hrf_ts.tsv', sep='\t').values[:, 1:]

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


def get_best_HRF(func_data, run_idx, PCs, PC_indicator, lambdas, events, TR, mask=False, work_dir=''):
	"""We fit each available HRF on each voxel for each run. We then take the median R2 for each
	HRF per voxel across runs. The best fitting HRF per voxel is returned.
	
	Arguments:
		func_data {[array]} -- Functional data TIME x VOXELS
		run_idx {[array]} -- list of run indices
		PCs {array}  -- Our 20 first noise PCs of size TIME x 20
		PC_indicator {array}  -- number of PCs to use for each voxel, size of len(voxels)
		lambdas {array}  -- lambda to use for each voxel, size of len(voxels)
		events {dataframe} -- Pandas DataFrame with the columns:  onset  duration  trial_type  run   full_onset
							  onset: the onset time for each event in relation to the run start
							  duration: duration of each event
							  trial_type: We assume singel trial, this should be np.arange(len(df))
							  run: a run indicator list (not used)
							  full_onset: onset times for each even in relation to the start of the FIRST run
		trial_types {array-like} -- list of length of events, condition identifier values for each onset
		TR {float} -- Repetition time for functional data

	
	Returns:
		[array] -- the indices for the best fitting HRF for each voxel
	"""

	ಠ_ಠ = ValueError('You need at least 2 runs')
	assert np.unique(run_idx).size > 1, ಠ_ಠ

	# how many HRFs do we ahve
	n_HRFs = HRF_KERNELS.shape[1]

	n_run = np.unique(run_idx).size

	# our HRF kernels are sampled at 10 hz
	oversampling = 10 * TR

	# get size of dimensions
	n_vols, n_voxels = func_data.shape

	# timing of samples
	frame_times = np.arange(n_vols) * TR

	# extract info from events DataFrame
	trial_type = events['trial_type'].values
	# we don't fit runs seperately, so we need the relative start from the first run
	onset = events['full_onset'].values
	duration = events['duration'].values
	modulations = np.ones(len(duration))

	# pre-allocate HRF x VOXEL
	r2s_hrf = np.zeros((n_HRFs, n_voxels))

	# loop over HRFs
	for i in tqdm(range(n_HRFs), desc=f'Calculating R2 for HRF:', file=tqdm_out):
		HRF_kernel = HRF_KERNELS[: ,i]

		# we have different number of PCs and Lambdas for each voxel,
		# so we need to fit the voxels separately
		vox = 0
		for vox in range(n_voxels):
			model = Ridge(lambd=lambdas[vox])

			# get regressor matrix for this HRF kernel
			X = get_regressor_matrix(HRF_kernel, frame_times, trial_type, onset,
									duration, modulations, oversampling=oversampling, min_onset=-25)
			
			# add the PC to the design matrix
			# add an intercept? yes for now
			# X = np.c_[X, PCs[:, :PC_indicator[vox]], np.ones(X.shape[0])]
			X = np.c_[X, np.ones(X.shape[0])]
			
			# indicate where the noise regressors (PCs) are in the design matrix
			apply_to = np.zeros(X.shape[1], dtype=bool)

			#apply_to[len(trial_type):len(trial_type)+PC_indicator[vox]] = True
			# fit data
			model.fit(X, func_data, apply_to)
			y_pred = model.predict(X)
			r2s_hrf[i, vox] = r2_score(func_data, y_pred)
	
	# pick best HRF for each voxel
	best_HRF = r2s_hrf.argmax(0)

	if mask:
		r2_img = masking.unmask(r2s_hrf, mask)
		f_out = os.path.join(work_dir, 'HRF_r2.nii.gz')
		r2_img.to_filename(f_out)

	return best_HRF


def optimize_signal_model(func_data, run_idx, PCs, PC_indicator, lambdas, events, TR, mask=False, work_dir=''):
	"""Fits the data with the preferred HRF of each voxel
	
	Arguments:
		func_data {[array]} -- Functional data SAMPLES x VOXELS
		run_idx {[array]} -- list of run indices
		events {dataframe} -- Pandas DataFrame with the columns:  onset  duration  trial_type  run   full_onset
							  onset: the onset time for each event in relation to the run start
							  duration: duration of each event
							  trial_type: We assume singel trial, this should be np.arange(len(df))
							  run: a run indicator list (not used)
							  full_onset: onset times for each even in relation to the start of the FIRST run
		TR {float} -- Repetition time for functional data

	
	Returns:
		fitted_brain {array} -- The data fit using preferred HRFs for each voxel, of size func_data.shape
		r2 {array} -- The R-square for each voxel
	"""
	ಠ_ಠ = ValueError('You need at least 2 runs')
	assert np.unique(run_idx).size > 1, ಠ_ಠ

	# get indices of the best HRF per voxel
	best_HRF = get_best_HRF(func_data, run_idx, PCs, PC_indicator,
							lambdas, events, TR, mask=False, work_dir='')
	
	# get specifics
	n_vols, n_voxels = func_data.shape
	frame_times = np.arange(n_vols) * TR

	# extract info from events DataFrame
	trial_type = events['trial_type'].values
	# we don't fit runs seperately, so we need the relative start from the first run
	onset = events['full_onset'].values
	duration = events['duration'].values
	modulations = np.ones(len(duration))

	oversampling = 10 * TR

	fitted_brain = np.zeros(func_data.shape)

	# loop over
	for vox in range(n_voxels):
		model = Ridge(lambd=lambdas[vox])
		HRF_kernel = best_HRF[vox]

		# get regressor matrix for this HRF kernel
		X = get_regressor_matrix(HRF_kernel, frame_times, trial_type, onset,
								duration, modulations, oversampling=oversampling, min_onset=-25)
		
		# add the PC to the design matrix
		# add an intercept? yes for now
		X = np.c_[X, PCs[PC_indicator[vox]], np.ones(X.shape[0])]
		
		# indicate where the noise regressors (PCs) are in the design matrix
		apply_to = np.zeros(X.shape[1], dtype=bool)
		apply_to[len(trial_type):trial_type+PC_indicator[vox]] = True
		# fit data
		model.fit(X, func_data, apply_to)
		fitted_brain[:, vox] = model.predict(X)
	
	r2 = r2_score(func_data, fitted_brain)

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
	# we also expect these inputs
	PCs = np.random.random([func_data.shape[0], 20]) # 20 PCs
	PC_indicator = np.random.choice(range(20), func_data.shape[1])
	lambdas = np.random.random(func_data.shape[1])

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
	
	event['full_onset'] = full_onset
	event['trial_type'] = np.arange(len(event)) # single trial

	"""
	Lets fit as we usually do
	"""
	event_data['onset'] = full_onset
	event_data['trial_type'] = np.arange(len(event)) # single trial
	import warnings
	warnings.simplefilter("ignore")
	X = make_design_matrix(
			frame_times, event_data, hrf_model='glover', drift_order=2, drift_model=None)

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
	best_HRF = get_best_HRF(func_data, run_idx, PCs, PC_indicator, lambdas, event, TR, mask=False, work_dir='')
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
