import os
import matplotlib
import os.path as op
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nilearn import masking
from joblib import Parallel, delayed
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
from nistats.first_level_model import run_glm
from nistats.contrasts import compute_contrast
from nistats.experimental_paradigm import check_events
from nistats.design_matrix import make_first_level_design_matrix
from nistats.hemodynamic_models import _sample_condition, _resample_regressor

from .preproc import hp_filter
from .logging import tqdm_ctm, tdesc
from .utils import get_run_data

here = op.dirname(__file__)
HRFS = pd.read_csv(op.join(here, 'data', 'hrf_ts.tsv'), sep='\t', index_col=0)


def _optimize_hrf(run, ddict, cfg, logger):
    """ Tries out 20 different Kay HRFs for a given run. """
    Y, _, events = get_run_data(ddict, run, func_type='denoised')
    nonzero = Y.sum(axis=0) != 0

    # Create 20 different design matrices
    ft = get_frame_times(ddict, cfg, Y)    
    dms = create_design_matrix(ddict['tr'], ft, events, hrf_model=cfg['hrf_model'])
    
    # Pre-allocate R2 array: 20 (hrfs) x K (voxels)
    r2 = np.zeros((HRFS.shape[1], Y.shape[1]))
    for i, X in enumerate(tqdm_ctm(dms, tdesc(f'Optimizing run {run+1}:'))):
        # Run GLM on nonzero voxels with current design matrix
        labels, results = run_glm(Y[:, nonzero], X.to_numpy(), noise_model='ols')
        r2[i, nonzero] = get_param_from_glm('r_square', labels, results, X, time_series=False)
                
    return r2


def _run_single_trial_model(run, best_hrf_idx, out_dir, ddict, cfg, logger):
    """ Fits a single trial model, possibly using an optimized HRF. """
    Y, _, events = get_run_data(ddict, run, func_type='denoised')
    ft = get_frame_times(ddict, cfg, Y)
    nonzero = Y.sum(axis=0) != 0
    
    # Which events are single trials (st)?
    st_idx = events['trial_type'].str.contains(cfg['single_trial_id'])
    st_names = events.loc[st_idx, 'trial_type']
    n_st = st_idx.sum()  # number of single trials

    cond_names = events.loc[~st_idx, 'trial_type'].unique()
    n_cond = cond_names.size + 1  # number of other conditions (+ icept)
    
    if best_hrf_idx.ndim > 1:  # run-specific HRF
        best_hrf_idx = best_hrf_idx[run, :]

    # Pre-allocate betas for single trial and conditions
    cond_betas = np.zeros((n_cond, Y.shape[1]))
    st_betas = np.zeros((n_st, Y.shape[1]))

    # Pre-allocate residuals
    residuals = np.zeros(Y.shape)

    if cfg['single_trial_model'] == 'lsa':
        preds = np.zeros(Y.shape)

    # Loop over unique HRF indices (0-20 probably)
    for hrf_idx in tqdm_ctm(np.unique(best_hrf_idx), tdesc(f'Final model run {run+1}:')):

        vox_idx = best_hrf_idx == hrf_idx
        vox_idx = np.logical_and(vox_idx, nonzero)  # create voxel mask

        if cfg['single_trial_model'] == 'lsa':
            X = create_design_matrix(ddict['tr'], ft, events, hrf_model=cfg['hrf_model'], hrf_idx=hrf_idx)
            #X.loc[:, :] = hp_filter(X.to_numpy(), ddict, cfg, logger, standardize='zscore')
            labels, results = run_glm(Y[:, vox_idx], X.to_numpy(), noise_model='ols')
            residuals[:, vox_idx] = get_param_from_glm('residuals', labels, results, X, time_series=True)
            preds[:, vox_idx] = get_param_from_glm('predicted', labels, results, X, time_series=True)

            for i, col in enumerate(st_names):
                cvec = np.zeros(X.shape[1])
                cvec[X.columns.tolist().index(col)] = 1
                st_betas[i, vox_idx] = compute_contrast(labels, results, con_val=cvec, contrast_type='t').effect_size()

            # uncorrelation (whiten patterns with covariance of design)
            # https://www.sciencedirect.com/science/article/pii/S1053811919310407
            if cfg['uncorrelation']:
                X_st = X.loc[:, X.columns.str.contains(cfg['single_trial_id'])].to_numpy()
                D = sqrtm(np.cov(X_st.T))
                st_betas = D @ st_betas

            for i, col in enumerate(cond_names):
                cvec = np.zeros(X.shape[1])
                cvec[X.columns.tolist().index(col)] = 1
                cond_betas[i, vox_idx] = compute_contrast(labels, results, con_val=cvec, contrast_type='t').effect_size()
        else:  # lss
            # Loop over single-trials
            for trial_nr, st_name in enumerate(st_names):
                events_cp = events.copy()  # copy original events dataframe
                # Set to-be-estimated trial to 'X', others to 'O', and create design matrix
                events_cp.loc[events_cp['trial_type'] == st_name, 'trial_type'] = 'X'
                events_cp.loc[events_cp['trial_type'].str.contains(cfg['single_trial_id']), 'trial_type'] = 'O'
                X = create_design_matrix(ddict['tr'], ft, events_cp, hrf_model=cfg['hrf_model'], hrf_idx=hrf_idx)

                # Run GLM and compute contrast for this single trial and other conditions
                labels, results = run_glm(Y[:, vox_idx], X.to_numpy(), noise_model='ols')
                residuals[:, vox_idx] += get_param_from_glm('residuals', labels, results, X, time_series=True)

                cvec = np.zeros(X.shape[1])
                cvec[X.columns.tolist().index('X')] = 1
                st_betas[trial_nr, vox_idx] = compute_contrast(labels, results, con_val=cvec, contrast_type='t').effect_size()
                
                for i, col in enumerate(cond_names):
                    cvec = np.zeros(X.shape[1])
                    cvec[X.columns.tolist().index(col)] = 1
                    cond_betas[i, vox_idx] += compute_contrast(labels, results, con_val=cvec, contrast_type='t').effect_size()

            # Because we estimated the betas for the other conditions len(st_names) times,
            # average them
            cond_betas /= st_names.size
            residuals /= st_names.size
     
    rdm = 1 - np.corrcoef(st_betas)
    plt.imshow(rdm)
    plt.savefig(f"rdm_run{run+1}_model-{cfg['hrf_model'].replace(' ', '')}_type-{cfg['single_trial_model']}.png")
    for i, name in enumerate(cond_names):    
        f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-{name}_beta.nii.gz')
        masking.unmask(cond_betas[i, :], ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-trial_beta.nii.gz')
    masking.unmask(st_betas, ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_residuals.nii.gz')
    masking.unmask(residuals, ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_predicted.nii.gz')
    masking.unmask(preds, ddict['mask']).to_filename(f_out)


def run_signal_processing(ddict, cfg, logger):
    """ Runs signal processing. """
    
    logger.info(f"Starting signal analysis.")
    out_dir = op.join(cfg['save_dir'], 'best')
    if not op.isdir(out_dir):
        os.makedirs(out_dir)
    
    if cfg['hrf_model'] == 'kay':  # try to optimize HRF selection
        # First, get R2 values for each HRF-based model (20 in total)
        r2 = Parallel(n_jobs=cfg['n_cpus'])(delayed(_optimize_hrf)
            (run, ddict, cfg, logger) for run in np.unique(ddict['run_idx'])
        )
        if cfg['save_all']:  # save to disk for inspection
            for i, this_r2 in enumerate(r2):  # hrf-wise r2 per run
                f_out = op.join(out_dir, cfg['f_base'] + f'_run-{i+1}_desc-hrf_r2.nii.gz')
                masking.unmask(this_r2, ddict['mask']).to_filename(f_out)

        # Stack into 3D array: M (runs) x 20 (hrfs) x K (voxels)
        r2 = np.stack(r2)

        if cfg['regularize_hrf_model']:  # same voxel-specific HRF for each run
            # IDEA: variance-weighted? So (r2_mean / r2_std).argmax(axis=0)?
            r2_median = np.median(r2, axis=0)  # median across runs

            # 1D array of size K (voxels) with best HRF index
            best_hrf_idx = r2_median.argmax(axis=0).astype(int)        

            if cfg['save_all']:
                f_out = op.join(out_dir, cfg['f_base'] + '_desc-hrf_r2.nii.gz')
                masking.unmask(r2_median, ddict['mask']).to_filename(f_out)
                # Save best index per voxel (and, optionally, per run)
                f_out = op.join(out_dir, cfg['f_base'] + '_desc-hrf_index.nii.gz')
                masking.unmask(best_hrf_idx, ddict['mask']).to_filename(f_out)
        else:  # specific HRF for each voxel and run (overfitting to the maxxxxx)
            best_hrf_idx = r2.argmax(axis=1)
    else:
        # bit of a hack
        best_hrf_idx = np.zeros(ddict['denoised_func'].shape[1])

    # Now, fit the single-trial models for real, using a voxel- (and possibly run-)
    # specific HRF or using a "fixed" one (if not regularize_hrf_model)
    Parallel(n_jobs=cfg['n_cpus'])(delayed(_run_single_trial_model)
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

    
def create_design_matrix(tr, frame_times, events, hrf_model='kay', hrf_idx=None):
    """ Creates a design matrix based on a HRF from Kendrick Kay's set
    or a default one from Nistats. """
    
    # Always oversample to milliseconds
    hrf_oversampling = 10
    # This is to keep oversampling consistent across hrf_models
    design_oversampling = tr / (0.1 / hrf_oversampling)
    
    if hrf_model != 'kay':
        return make_first_level_design_matrix(
            frame_times, events, drift_model=None, min_onset=0,
            oversampling=design_oversampling, hrf_model=hrf_model
        )

    if hrf_model == 'kay':

        # Note: Kendrick's HRFs are defined at 0.1 sec resolution
        t_hrf = HRFS.index.copy()

        # Resample to msec resolution
        f = interp1d(t_hrf, HRFS, axis=0)
        t_high = np.linspace(0, 50, num=HRFS.shape[0]*hrf_oversampling, endpoint=True)
        hrfs_hr = f(t_high).T  # hr = high resolution

        if hrf_idx is None:
            to_iter = range(HRFS.shape[1])
        else:
            to_iter = [hrf_idx]

        # dms will store all design matrices
        dms = []
        for hrf_idx in to_iter:
            hrf = hrfs_hr[:, hrf_idx]
            
            # To match the design oversampling, do it relative to tr
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
                # Create high resolution regressor/frame times
                hr_regressor, hr_frame_times = _sample_condition(
                    exp_condition, frame_times, design_oversampling, 0
                )
                
                # Convolve with HRF and downsample
                conv_reg = np.convolve(hr_regressor, hrf)[:hr_regressor.size]
                f = interp1d(hr_frame_times, conv_reg)
                X[:, i] = f(frame_times).T
            
            # Store in dms
            dms.append(pd.DataFrame(X, columns=uniq_trial_types, index=frame_times))

        if len(dms) == 1:
            dms = dms[0]

        return dms 


def get_frame_times(ddict, cfg, Y):
    """ Computes frame times for a particular time series (and TR). """
    tr = ddict['tr']
    n_vol = Y.shape[0]
    st_ref = cfg['slice_time_ref']
    ft = np.linspace(st_ref * tr, n_vol * tr + st_ref * tr, n_vol, endpoint=False)
    return ft
