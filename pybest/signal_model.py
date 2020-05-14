import warnings
import os
import os.path as op
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from nilearn import signal, masking
from sklearn.linear_model import Ridge
from nistats.first_level_model import run_glm
from nistats.experimental_paradigm import check_events
from nistats.reporting import plot_design_matrix
from nistats.hemodynamic_models import _sample_condition, _resample_regressor
from nistats.reporting import plot_design_matrix
from nistats.contrasts import compute_contrast
from scipy.linalg import sqrtm
from .preproc import hp_filter
from .utils import get_run_data, yield_uniq_params, tqdm_ctm, tdesc


here = op.dirname(__file__)
HRFS = pd.read_csv(op.join(here, 'data', 'hrf_ts.tsv'), sep='\t', index_col=0)


def _run_hrf_optimization_parallel(run, ddict, cfg, logger):
    """ Tries out 20 different HRFs for a given run. """
    Y, _, events = get_run_data(ddict, run, func_type='denoised')
    nonzero = Y.sum(axis=0) != 0
    ft = get_frame_times(ddict, cfg, Y)
    
    # r2 = 20 (hrfs) x K (voxels)
    r2 = np.zeros((HRFS.shape[1], Y.shape[1]))
    for hrf_idx in tqdm_ctm(range(HRFS.shape[1]), tdesc(f'Optimizing run {run+1}:')):
        # Make single-trial (LSA) design matrix
        X = create_design_matrix(hrf_idx, ddict['tr'], ft, events)
        st_idx = X.columns.str.contains(cfg['single_trial_id'])

        # Fit intercept only model
        X['intercept'] = X.loc[:, st_idx].sum(axis=1)
        st_idx = np.r_[st_idx, False]
        X_icp = X.iloc[:, ~st_idx].copy()  # remove single trials
        
        # Orthogonalize high-pass filter to intercept model (also normalizes) and fit
        X_icp.iloc[:, :] = hp_filter(X_icp.to_numpy(), ddict, cfg, logger)
        labels, results = run_glm(Y[:, nonzero], X_icp.to_numpy(), noise_model='ols')  # change to ar1

        # Fit trial model on residuals of intercept model
        Y_resids = get_param_from_glm('residuals', labels, results, X_icp, time_series=True)
        Y_resids = signal.clean(Y_resids, detrend=False, standardize='zscore')
        X_trial = X.loc[:, st_idx].copy()  # stupid SettingWithCopyWarning
        labels, results = run_glm(Y_resids, X_trial.to_numpy(), noise_model='ols')
        r2[hrf_idx, nonzero] = get_param_from_glm('r_square', labels, results, X_trial, time_series=False)
                
    return r2


def _run_signal_model_parallel(run, best_hrf_idx, out_dir, ddict, cfg, logger):

    Y, _, events = get_run_data(ddict, run, func_type='denoised')
    ft = get_frame_times(ddict, cfg, Y)

    # Saving everything for now (remove at some point)
    residuals_icept_model = np.zeros_like(Y)
    residuals_trial_model = np.zeros_like(Y)
    preds_icept_model = np.zeros_like(Y)
    preds_trial_model = np.zeros_like(Y)
    
    r2_icept_model = np.zeros(Y.shape[1])
    r2_trial_model = np.zeros(Y.shape[1])
    
    st_idx = events['trial_type'].str.contains(cfg['single_trial_id'])
    n_st = st_idx.sum()
    n_cond = events.loc[~st_idx, 'trial_type'].unique().size + 1
    
    #st_onsets = np.round(events.loc[st_idx, 'onset'].to_numpy()).astype(int)
    #other_onsets = np.round(events.loc[~st_idx, 'onset'].to_numpy()).astype(int)
    #stim_vec = np.zeros(int(Y.shape[0] * ddict['tr']))
    #stim_vec[st_onsets] = 2
    #stim_vec[other_onsets] = 1
    #np.savetxt(op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-stim_onsets.txt'), stim_vec)

    if best_hrf_idx.ndim > 1:
        best_hrf_idx = best_hrf_idx[run, :]

    # Just for visualization
    X = create_design_matrix(10, ddict['tr'], ft, events)
    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-design_matrix.png')
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_design_matrix(X, output_file=f_out, ax=ax)

    cond_betas = np.zeros((n_cond, Y.shape[1]))
    trial_betas = np.zeros((n_st, Y.shape[1]))
    for hrf_idx in tqdm_ctm(np.unique(best_hrf_idx), tdesc(f'Final model run {run+1}:')):
        
        X = create_design_matrix(hrf_idx, ddict['tr'], ft, events)
        vox_idx = best_hrf_idx == hrf_idx
        nonzero = Y.sum(axis=0) != 0
        vox_idx = np.logical_and(vox_idx, nonzero)

        st_idx = X.columns.str.contains(cfg['single_trial_id'])

        # Fit intercept only model
        X['intercept'] = X.loc[:, st_idx].sum(axis=1)
        X_icp = X.iloc[:, ~np.r_[st_idx, False]].copy()  # remove single trials

        # Orthogonalize high-pass filter to intercept model (also normalizes) and fit
        X_icp.iloc[:, :] = hp_filter(X_icp.to_numpy(), ddict, cfg, logger)
        labels, results = run_glm(Y[:, vox_idx], X_icp.to_numpy(), noise_model='ols')  # change to ar1

        # Store average response for inspection
        for i, col in enumerate(X_icp.columns):
            cvec = np.zeros(X_icp.shape[1])
            cvec[i] = 1
            cond_betas[i, vox_idx] = compute_contrast(labels, results, con_val=cvec, contrast_type='t').effect_size()
        
        preds_icept_model[:, vox_idx] = get_param_from_glm('predicted', labels, results, X_icp, time_series=True)
        r2_icept_model[vox_idx] = get_param_from_glm('r_square', labels, results, X_icp, time_series=False)

        # Fit trial model on residuals of intercept model
        residuals_icept_model[:, vox_idx] = get_param_from_glm('residuals', labels, results, X_icp, time_series=True)
        X_trial = X.iloc[:, st_idx].copy()  # stupid SettingWithCopyWarning
        labels, results = run_glm(residuals_icept_model[:, vox_idx], X_trial.to_numpy(), noise_model='ols')

        preds_trial_model[:, vox_idx] = get_param_from_glm('predicted', labels, results, X_trial, time_series=True)    
        residuals_trial_model[:, vox_idx] = get_param_from_glm('residuals', labels, results, X_trial, time_series=True)
        r2_trial_model[vox_idx] = get_param_from_glm('r_square', labels, results, X_trial, time_series=False)
        for i, col in enumerate(X_trial.columns):
            cvec = np.zeros(X_trial.shape[1])
            cvec[X_trial.columns.tolist().index(col)] = 1
            trial_betas[i, vox_idx] = compute_contrast(labels, results, con_val=cvec, contrast_type='t').effect_size()

        #D = sqrtm(np.linalg.inv(np.cov(X_trial.to_numpy().T)))
        #trial_betas = D @ trial_betas

    for i, name in enumerate(X_icp.columns):    
        f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-{name}_beta.nii.gz')
        masking.unmask(cond_betas[i, :], ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-trial_beta.nii.gz')
    masking.unmask(trial_betas, ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-intercept_r2.nii.gz')
    masking.unmask(r2_icept_model, ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-trial_r2.nii.gz')
    masking.unmask(r2_trial_model, ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-intercept_predicted.nii.gz')
    masking.unmask(preds_icept_model, ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-trial_predicted.nii.gz')
    masking.unmask(preds_trial_model, ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-intercept_residuals.nii.gz')
    masking.unmask(residuals_icept_model, ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-trial_residuals.nii.gz')
    masking.unmask(residuals_trial_model, ddict['mask']).to_filename(f_out)


def run_signal_processing(ddict, cfg, logger):
    """ Runs signal processing. """
    
    logger.info(f"Starting signal analysis.")
    sub, ses, task = cfg['sub'], cfg['ses'], cfg['task']
    out_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'best')
    if not op.isdir(out_dir):
        os.makedirs(out_dir)
    
    # First, get R2 values for each HRF-based model (20 in total)
    r2 = Parallel(n_jobs=cfg['n_cpus'])(delayed(_run_hrf_optimization_parallel)
        (run, ddict, cfg, logger) for run in np.unique(ddict['run_idx'])
    )
    for i, this_r2 in enumerate(r2):  # save to disk for inspection
        f_out = op.join(out_dir, cfg['f_base'] + f'_run-{i+1}_desc-hrf_r2.nii.gz')
        masking.unmask(this_r2, ddict['mask']).to_filename(f_out)

    # runs x 20 (hrfs) x K (voxels)
    r2 = np.stack(r2)

    if cfg['regularize_hrf_model']:  # same voxel-specific HRF for each run
        # IDEA: variance-weighted? So (r2_mean / r2_std).argmax(axis=0)?
        r2_median = np.median(r2, axis=0)  # median across runs
        f_out = op.join(out_dir, cfg['f_base'] + '_desc-hrf_r2.nii.gz')
        masking.unmask(r2_median, ddict['mask']).to_filename(f_out)
        best_hrf_idx = r2_median.argmax(axis=0).astype(int)        
    else:  # specific HRF for each voxel and run (overfitting to the maxxxxx)
        best_hrf_idx = r2.argmax(axis=1)
    
    #best_hrf_idx = np.ones(ddict['denoised_func'].shape[1], dtype=int) * 10

    # Save best index per voxel (and, optionally, per run)
    f_out = op.join(out_dir, cfg['f_base'] + '_desc-hrf_index.nii.gz')
    masking.unmask(best_hrf_idx, ddict['mask']).to_filename(f_out)

    # Refit (computationally inefficient, but storing each previous model fit
    # uses too much memory)
    Parallel(n_jobs=cfg['n_cpus'])(delayed(_run_signal_model_parallel)
        (run, best_hrf_idx, out_dir, ddict, cfg, logger) for run in np.unique(ddict['run_idx'])
    )


def get_param_from_glm(name, labels, results, dm, time_series=False):
    """ Get parameters from a fitted nistats GLM. """
    if time_series:
        data = np.zeros((dm.shape[0], labels.size))
    else:
        data = np.zeros_like(labels)

    for lab in np.unique(labels):
        data[..., labels == lab] = getattr(results[lab], name)
    
    return data



def create_design_matrix(hrf_idx, tr, frame_times, events):
    """ Creates a design matrix based on a HRF from Kendrick Kay's set. """
    # Always oversample to milliseconds
    hrf_oversampling = 10

    # Note: Kendrick's HRFs are defined at 0.1 sec resolution
    t_hrf = HRFS.index.copy()
    hrf = HRFS.iloc[:, hrf_idx].to_numpy()

    # Oversample to milliseconds
    f = interp1d(t_hrf, hrf)
    t_high = np.linspace(0, 50, num=hrf.size*hrf_oversampling, endpoint=True) 
    hrf = f(t_high).T

    # To match the design oversampling, do it relative to tr
    design_oversampling = tr / (0.1 / hrf_oversampling)
    trial_type, onset, duration, modulation = check_events(events)

    # Pre-allocate design matrix; note: columns are alphabetically sorted
    X = np.zeros((frame_times.size, np.unique(trial_type).size))
    uniq_trial_types = np.unique(trial_type)
    for i, condition in enumerate(uniq_trial_types):
        condition_mask = (trial_type == condition)
        exp_condition = (
            onset[condition_mask],
            duration[condition_mask],
            modulation[condition_mask]
        )
        hr_regressor, hr_frame_times = _sample_condition(exp_condition, frame_times, design_oversampling, 0)
        # Convolve with HRF and downsample
        conv_reg = np.convolve(hr_regressor, hrf)[:hr_regressor.size]
        f = interp1d(hr_frame_times, conv_reg)
        X[:, i] = f(frame_times).T

    return pd.DataFrame(X, columns=uniq_trial_types, index=frame_times) 


def get_frame_times(ddict, cfg, Y):
    """ Computes frame times for a particular time series (and TR). """
    tr = ddict['tr']
    n_vol = Y.shape[0]
    st_ref = cfg['slice_time_ref']
    ft = np.linspace(st_ref * tr, n_vol * tr + st_ref * tr, n_vol, endpoint=False)
    return ft
