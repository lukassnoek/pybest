import os
import matplotlib
import os.path as op
import numpy as np
import pandas as pd
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

from nilearn import masking
from joblib import Parallel, delayed
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
from scipy.stats.mstats import rankdata
from sklearn.linear_model import LinearRegression
from nilearn.stats.first_level_model import run_glm
from nilearn.stats.contrasts import compute_contrast
from nilearn.stats.first_level_model.experimental_paradigm import check_events
from nilearn.stats.first_level_model.design_matrix import make_first_level_design_matrix
from nilearn.stats.first_level_model.hemodynamic_models import _sample_condition, _resample_regressor

from .constants import HRFS_HR
from .preproc import hp_filter
from .logging import tqdm_ctm, tdesc
from .utils import get_run_data, get_frame_times, get_param_from_glm, yield_glm_results


STATS = dict(beta='effect_size', zscore='z_score')


def _optimize_hrf(run, ddict, cfg, logger):
    """ Tries out 20 different Kay HRFs for a given run. """
    Y, conf, events = get_run_data(ddict, run, func_type='denoised')
    nonzero = ~np.all(np.isclose(Y, 0.), axis=0)

    # Create 20 different design matrices
    ft = get_frame_times(ddict, cfg, Y)    
    dms = create_design_matrix(ddict['trs'][run], ft, events, hrf_model=cfg['hrf_model'])
    
    # Pre-allocate R2 array: 20 (hrfs) x K (voxels)
    r2 = np.zeros((20, Y.shape[1]))
    for i, X in enumerate(tqdm_ctm(dms, tdesc(f'Optimizing run {run+1}:'))):
        # High-pass design matrix (except intercept) and remove noise
        X.iloc[:, :-1] = hp_filter(X.iloc[:, :-1].to_numpy(), ddict['trs'][run], ddict, cfg, logger)

        for this_vox_idx, _, labels, results in yield_glm_results(nonzero, Y, X, conf, run, ddict, cfg, noise_model='ols'):
            r2[i, this_vox_idx] = get_param_from_glm('r_square', labels, results, X, time_series=False)

    r2 = rankdata(r2, axis=0)
    return r2


def _run_single_trial_model(run, best_hrf_idx, out_dir, ddict, cfg, logger):
    """ Fits a single trial model, possibly using an optimized HRF. """
    Y, conf, events = get_run_data(ddict, run, func_type='denoised')
    ft = get_frame_times(ddict, cfg, Y)
    nonzero = ~np.all(np.isclose(Y, 0.), axis=0)

    # Which events are single trials (st)?
    st_idx = events['trial_type'].str.contains(cfg['single_trial_id'])
    st_names = events.loc[st_idx, 'trial_type']
    cond_names = events.loc[~st_idx, 'trial_type'].unique().tolist() + ['constant'] 

    if best_hrf_idx.ndim > 1:  # run-specific HRF
        best_hrf_idx = best_hrf_idx[run, :]

    # Pre-allocate residuals and r2
    residuals = np.zeros(Y.shape)
    r2 = np.zeros(Y.shape[1])
    patterns = np.zeros((events['trial_type'].unique().size + 1, Y.shape[1]))

    if cfg['single_trial_model'] == 'lsa':
        preds = np.zeros(Y.shape)
        st_icept = np.zeros(Y.shape[1])
    
    # Loop over unique HRF indices (0-20 probably)
    for hrf_idx in tqdm_ctm(np.unique(best_hrf_idx), tdesc(f'Final model run {run+1}:')):
        # Create voxel mask (nonzero ^ hrf index)
        vox_idx = best_hrf_idx == hrf_idx
        vox_idx = np.logical_and(vox_idx, nonzero)
        
        if cfg['single_trial_model'] == 'lsa':  # least-squares all
            # Get current design matrix, hp-filter, and start noise loop
            X = create_design_matrix(ddict['trs'][run], ft, events, hrf_model=cfg['hrf_model'], hrf_idx=hrf_idx)
            X.iloc[:, :-1] = hp_filter(X.iloc[:, :-1].to_numpy(), ddict['trs'][run], ddict, cfg, logger)
            st_idx_x = X.columns.str.contains(cfg['single_trial_id'])

            # Loop across unique n comps
            for this_vox_idx, this_X, labels, results in yield_glm_results(vox_idx, Y, X, conf, run, ddict, cfg, noise_model=cfg['single_trial_noise_model']):                
                # Extract residuals, predictions, and r2
                residuals[:, this_vox_idx] = get_param_from_glm('residuals', labels, results, this_X, time_series=True)
                preds[:, this_vox_idx] = get_param_from_glm('predicted', labels, results, this_X, time_series=True)
                r2[this_vox_idx] = get_param_from_glm('r_square', labels, results, this_X, time_series=False)

                # Loop over columns to extract parameters/zscores
                for i, col in enumerate(this_X.columns):
                    cvec = np.zeros(this_X.shape[1])
                    cvec[this_X.columns.tolist().index(col)] = 1
                    con = compute_contrast(labels, results, con_val=cvec, contrast_type='t')
                    patterns[i, this_vox_idx] = getattr(con, STATS[cfg['pattern_units']])()

                # Get "intercept" (average effect) of single-trials
                cvec = np.zeros(this_X.shape[1])
                cvec[st_idx_x] = 1  # set all single-trial columns to 1 in contrast-vec
                con = compute_contrast(labels, results, con_val=cvec, contrast_type='t')
                st_icept[this_vox_idx] = getattr(con, STATS[cfg['pattern_units']])()

                # uncorrelation (whiten patterns with covariance of design)
                # https://www.sciencedirect.com/science/article/pii/S1053811919310407
                if cfg['uncorrelation']:
                    X_ = this_X.iloc[:, :-1].to_numpy()
                    D = sqrtm(np.cov(X_.T))
                    # Do not uncorrelate constant
                    patterns[:-1, this_vox_idx] = D @ patterns[:-1, this_vox_idx]
        else:  # If not LSA, do LSS
            # Loop over single-trials
            for trial_nr, st_name in enumerate(st_names):
                events_cp = events.copy()  # copy original events dataframe
                # Set to-be-estimated trial to 'X', others to 'O', and create design matrix
                events_cp.loc[events_cp['trial_type'] == st_name, 'trial_type'] = 'X'
                events_cp.loc[events_cp['trial_type'].str.contains(cfg['single_trial_id']), 'trial_type'] = 'O'
                X = create_design_matrix(ddict['trs'][run], ft, events_cp, hrf_model=cfg['hrf_model'], hrf_idx=hrf_idx)
                X.iloc[:, :-1] = hp_filter(X.iloc[:, :-1].to_numpy(), ddict['trs'][run], ddict, cfg, logger)
                # Loop across unique n comps
                for this_vox_idx, this_X, labels, results in yield_glm_results(vox_idx, Y, X, conf, run, ddict, cfg, noise_model=cfg['single_trial_noise_model']):
                    residuals[:, this_vox_idx] += get_param_from_glm('residuals', labels, results, this_X, time_series=True)
                    r2[this_vox_idx] += get_param_from_glm('r_square', labels, results, this_X, time_series=False)

                    cvec = np.zeros(this_X.shape[1])
                    cvec[this_X.columns.tolist().index('X')] = 1
                    patterns[trial_nr, this_vox_idx] = compute_contrast(labels, results, con_val=cvec, contrast_type='t').effect_size()
                    
                    for i, col in enumerate(cond_names):
                        cvec = np.zeros(this_X.shape[1])
                        cvec[this_X.columns.tolist().index(col)] = 1
                        patterns[len(st_names)+i, this_vox_idx] += compute_contrast(labels, results, con_val=cvec, contrast_type='t').effect_size()

            # Because we estimated the betas for the other conditions len(st_names) times,
            # average them by dividing them by the number of trials
            patterns[len(st_names):, nonzero] /= st_names.size
            residuals[:, nonzero] /= st_names.size
            r2[nonzero] /= st_names.size

            st_idx_x = np.zeros(patterns.shape[0], dtype=bool)  
            st_idx_x[:len(st_names)] = True
            st_icept = patterns[:len(st_names), :].mean(axis=0)
    
    st_patterns = patterns[st_idx_x, :]
    cond_patterns = patterns[~st_idx_x, :]

    unit = cfg['pattern_units']
    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-stimicept_{unit}.nii.gz')
    masking.unmask(st_icept, ddict['mask']).to_filename(f_out)

    #rdm = 1 - np.corrcoef(st_patterns)
    #plt.imshow(rdm)
    #plt.savefig(f"rdm_run{run+1}_model-{cfg['hrf_model'].replace(' ', '')}_type-{cfg['single_trial_model']}.png")
    #plt.close()
    for i, name in enumerate(cond_names):    
        f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-{name}_{unit}.nii.gz')
        masking.unmask(cond_patterns[i, :], ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-trial_{unit}.nii.gz')
    masking.unmask(st_patterns, ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_residuals.nii.gz')
    masking.unmask(residuals, ddict['mask']).to_filename(f_out)

    if cfg['single_trial_model'] == 'lsa':
        f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_predicted.nii.gz')
        masking.unmask(preds, ddict['mask']).to_filename(f_out)

    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_r2.nii.gz')
    masking.unmask(r2, ddict['mask']).to_filename(f_out)


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
            best_hrf_idx = r2.argmax(axis=1).astype(int)
    else:
        # bit of a hack: set all voxels to the same HRF (index: 0)
        best_hrf_idx = np.zeros(ddict['denoised_func'].shape[1]).astype(int)
    
    #best_hrf_idx = np.random.randint(20, size=ddict['denoised_func'].shape[1])
    
    # Now, fit the single-trial models for real, using a voxel- (and possibly run-)
    # specific HRF or using a "fixed" one (if not regularize_hrf_model)
    Parallel(n_jobs=cfg['n_cpus'])(delayed(_run_single_trial_model)
        (run, best_hrf_idx, out_dir, ddict, cfg, logger) for run in np.unique(ddict['run_idx'])
    )

 
def create_design_matrix(tr, frame_times, events, hrf_model='kay', hrf_idx=None):
    """ Creates a design matrix based on a HRF from Kendrick Kay's set
    or a default one from Nistats. """
    
    # This is to keep oversampling consistent across hrf_models
    hrf_oversampling = 10
    design_oversampling = tr / (0.1 / hrf_oversampling)

    if hrf_model != 'kay':
        return make_first_level_design_matrix(
            frame_times, events, drift_model=None, min_onset=0,
            oversampling=design_oversampling, hrf_model=hrf_model
        )

    if hrf_model == 'kay':
        
        if hrf_idx is None:
            to_iter = range(HRFS_HR.shape[1])
        else:
            to_iter = [hrf_idx]

        # dms will store all design matrices
        dms = []
        for hrf_idx in to_iter:
            hrf = HRFS_HR[:, hrf_idx]
            
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
            X /= X.max(axis=0)  # rescale to max = 1
            dm = pd.DataFrame(X, columns=uniq_trial_types, index=frame_times)
            dm['constant'] = 1
            dms.append(dm)

        if len(dms) == 1:
            dms = dms[0]

        return dms 
