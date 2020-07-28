import os
import matplotlib.pyplot as plt
import os.path as op
import numpy as np
import pandas as pd
from tqdm import tqdm
from nilearn import masking
from joblib import Parallel, delayed
from scipy import stats
from scipy.linalg import sqrtm
from scipy.stats.mstats import rankdata
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from nilearn._utils.glm import z_score
from nilearn.glm.first_level import run_glm
from nilearn.glm.contrasts import compute_contrast, expression_to_contrast_vector
from .logging import tqdm_ctm, tdesc
from .utils import get_run_data, get_frame_times, get_param_from_glm, yield_glm_results
from .utils import hp_filter, create_design_matrix, save_data, custom_clean
from .constants import STATS
from .models import cross_val_r2


def run_signal_processing(ddict, cfg, logger):
    """ Runs signal processing using either single-trail type models (LSA/LSS)
    or a cross-validated GLMdenoise type model. """
    
    if cfg['skip_signalproc']:
        logger.warn("Skipping signal processing (because of --skip-signalproc)")
        return None

    logger.info(f"Starting signal analysis")

    if cfg['signalproc_type'] == 'single-trial':
        _run_single_trial_model(ddict, cfg, logger)
    else:  # GLMdenoise style condition estimation
        _run_glmdenoise_model(ddict, cfg, logger)


def _run_single_trial_model(ddict, cfg, logger):
    """ Single-trial estimation. """

    logger.info(f"Running single-trial estimation")

    n_runs = np.unique(ddict['run_idx']).size
    K = ddict['denoised_func'].shape[1]

    if cfg['hrf_model'] == 'kay':  # try to optimize HRF selection
        logger.info(f"Going to optimize the HRF (using Kay's 20-HRF basis set)")
        # First, get R2 values for each HRF-based model (20 in total)
        # r2: list (n_runs) of 2D (20 x voxels) arrays
        r2 = Parallel(n_jobs=cfg['n_cpus'])(delayed(_optimize_hrf_within)
            (run, ddict, cfg, logger) for run in range(n_runs)
        )
        if cfg['save_all']:  # save to disk for inspection
            for run, this_r2 in enumerate(r2):  # hrf-wise r2 per run
                save_data(this_r2, cfg, ddict, par_dir='best', run=run+1,
                          desc='hrf', dtype='r2')

        # Stack into 3D array: M (runs) x 20 (hrfs) x K (voxels)
        r2 = np.stack(r2)

        if cfg['regularize_hrf_model']:  # same voxel-specific HRF for each run
            logger.info("Regularizing HRF model")
            # IDEA: variance-weighted? So (r2_mean / r2_std).argmax(axis=0)?
            # IDEA: rank-transform per run
            r2_median = np.median(r2 - r2.mean(axis=0), axis=0)  # median across runs

            # 1D array of size K (voxels) with best HRF index
            best_hrf_idx = r2_median.argmax(axis=0).astype(int)

            if cfg['save_all']:  # save per-run statistics
                save_data(r2_median, cfg, ddict, par_dir='best', run=None, desc='hrf', dtype='r2')
                save_data(best_hrf_idx, cfg, ddict, par_dir='best', run=None, desc='opt', dtype='hrf')
        else:  # specific HRF for each voxel and run (2D array: runs x voxels)
            best_hrf_idx = r2.argmax(axis=1).astype(int)
    else:
        # bit of a hack: set all voxels to the same HRF (index: 0)
        best_hrf_idx = np.zeros(K).astype(int)
    
    # Now, fit the single-trial models for real, using a voxel- (and possibly run-)
    # specific HRF or using a "fixed" one (if not --regularize-hrf-model)
    Parallel(n_jobs=cfg['n_cpus'])(delayed(_run_single_trial_model_parallel)
        (run, best_hrf_idx, ddict, cfg, logger) for run in range(n_runs)
    )


def _optimize_hrf_within(run, ddict, cfg, logger):
    """ Tries out 20 different Kay HRFs for a given run. """
    Y, conf, events = get_run_data(ddict, run, func_type='denoised')
    nonzero = ~np.all(np.isclose(Y, 0.), axis=0)

    # Create 20 different design matrices
    tr = ddict['trs'][run]
    ft = get_frame_times(tr, ddict, cfg, Y)    
    dms = create_design_matrix(tr, ft, events, hrf_model=cfg['hrf_model'])
    
    # Pre-allocate R2 array: 20 (hrfs) x K (voxels)
    r2 = np.zeros((20, Y.shape[1]))
    for i, X in enumerate(tqdm_ctm(dms, tdesc(f'Optimizing run {run+1}:'))):
        # Note: GLM is fitted inside yield_glm_results function
        for out in yield_glm_results(nonzero, Y, X, conf, run, ddict, cfg):
            this_vox_idx, _, labels, results = out
            r2[i, this_vox_idx] = get_param_from_glm('r_square', labels, results, X, time_series=False)

    #r2 = rankdata(r2, axis=0)
    return r2


def _run_single_trial_model_parallel(run, best_hrf_idx, ddict, cfg, logger):
    """ Fits a single trial model, possibly using an optimized HRF. """
    Y, conf, events = get_run_data(ddict, run, func_type='denoised')
    K = Y.shape[1]
    tr = ddict['trs'][run]

    # ft = frame times (Nilearn lingo)
    ft = get_frame_times(tr, ddict, cfg, Y)
    nonzero = ~np.all(np.isclose(Y, 0.), axis=0)

    # Which events are single trials (st)?
    if cfg['single_trial_id'] is None:
        st_idx = np.zeros(events.shape[0]).astype(bool)
    else:
        st_idx = events['trial_type'].str.contains(cfg['single_trial_id'])
    
    # What are the names of the single-trials (st) and other conditions (cond)?
    st_names = events.loc[st_idx, 'trial_type']
    cond_names = events.loc[~st_idx, 'trial_type'].unique().tolist()
    if cfg['single_trial_id'] is not None:
        cond_names += ['unmodstim']

    if best_hrf_idx.ndim > 1:  # run-specific HRF
        best_hrf_idx = best_hrf_idx[run, :]

    # Pre-allocate residuals and r2
    residuals = np.zeros(Y.shape)
    r2 = np.zeros(Y.shape[1])
    # Note to self: the +1 is from the stim_unmod regressor
    patterns = np.zeros((len(st_names) + len(cond_names), K))
    
    if cfg['single_trial_model'] == 'lsa':
        preds = np.zeros(Y.shape)
        preds_icept = np.zeros(Y.shape)
        #st_icept = np.zeros(K)
    
    if cfg['contrast'] is not None:
        # ccon = custom contrast
        ccon = np.zeros(K)

    # Loop over unique HRF indices (0-20 probably)
    for hrf_idx in tqdm_ctm(np.unique(best_hrf_idx), tdesc(f'Final model run {run+1}:')):
        # Create voxel mask (nonzero ^ hrf index)
        vox_idx = best_hrf_idx == hrf_idx
        vox_idx = np.logical_and(vox_idx, nonzero)

        if cfg['single_trial_model'] == 'lsa':  # least-squares all
            # Get current design matrix, hp-filter, and start noise loop
            X = create_design_matrix(tr, ft, events, hrf_model=cfg['hrf_model'], hrf_idx=hrf_idx)
            X = X.drop('constant', axis=1)

            if cfg['single_trial_id'] is not None:
                st_idx_x = X.columns.str.contains(cfg['single_trial_id'])
                # Add "unmodulated stimulus" regressor
                X['unmodstim'] = X.loc[:, st_idx_x].sum(axis=1)
                st_idx_x = np.r_[st_idx_x, False]

            # Loop across unique n comps
            for out in yield_glm_results(vox_idx, Y, X, conf, run, ddict, cfg):                
                this_vox_idx, this_X, labels, results = out
                # Extract residuals, predictions, and r2
                residuals[:, this_vox_idx] = get_param_from_glm('residuals', labels, results, this_X, time_series=True)
                preds[:, this_vox_idx] = get_param_from_glm('predicted', labels, results, this_X, time_series=True)
                r2[this_vox_idx] = get_param_from_glm('r_square', labels, results, this_X, time_series=False)
                beta = get_param_from_glm('theta', labels, results, this_X, time_series=False, predictors=True)
                preds_icept[:, this_vox_idx] = this_X.iloc[:, -2:].to_numpy() @ beta[-2:, :]

                # Loop over columns to extract parameters/zscores
                for i, col in enumerate(this_X.columns):
                    if col == 'constant':
                        continue

                    cvec = np.zeros(this_X.shape[1])
                    cvec[this_X.columns.tolist().index(col)] = 1
                    con = compute_contrast(labels, results, con_val=cvec, contrast_type='t')
                    patterns[i, this_vox_idx] = getattr(con, STATS[cfg['pattern_units']])()

                ## Get "intercept" (average effect) of single-trials
                #if cfg['single_trial_id'] is not None:
                #    cvec = np.zeros(this_X.shape[1])
                #    cvec[st_idx_x] = 1  # set all single-trial columns to 1 in contrast-vec
                #    con = compute_contrast(labels, results, con_val=cvec, contrast_type='t')
                #    st_icept[this_vox_idx] = getattr(con, STATS[cfg['pattern_units']])()

                # Evaluate "custom contrast" if there is any
                if cfg['contrast'] is not None:
                    cvec = expression_to_contrast_vector(cfg['contrast'], this_X.columns.tolist())
                    con = compute_contrast(labels, results, con_val=cvec, contrast_type='t')
                    ccon[this_vox_idx] = getattr(con, STATS[cfg['pattern_units']])()

                # uncorrelation (whiten patterns with covariance of design)
                # https://www.sciencedirect.com/science/article/pii/S1053811919310407
                if cfg['uncorrelation']:
                    if 'constant' in this_X.columns:
                        this_X = this_X.drop('constant', axis=1)

                    #if 'unmodstim' in this_X.columns:
                    #    this_X = this_X.drop('unmodstim', axis=1)

                    X_ = this_X.to_numpy()
                    D = sqrtm(np.corrcoef(X_.T))
                    patterns[:, this_vox_idx] = D @ patterns[:, this_vox_idx]

        else:  # Least-squares separate (LSS)
            
            if len(st_names) == 0:
                raise ValueError("Probably not wise to do LSS without single trials")

            # Loop across single-trials
            for trial_nr, st_name in enumerate(st_names):
                events_cp = events.copy()  # copy original events dataframe
                # Set to-be-estimated trial to 'X', others to 'O', and create design matrix
                events_cp.loc[events_cp['trial_type'] == st_name, 'trial_type'] = 'X'
                events_cp.loc[events_cp['trial_type'].str.contains(cfg['single_trial_id']), 'trial_type'] = 'O'
                X = create_design_matrix(tr, ft, events_cp, hrf_model=cfg['hrf_model'], hrf_idx=hrf_idx)
                X = X.drop('constant', axis=1)

                # Loop across unique n comps
                for out in yield_glm_results(vox_idx, Y, X, conf, run, ddict, cfg):
                    this_vox_idx, this_X, labels, results = out
                    # Store residuals, r2
                    residuals[:, this_vox_idx] += get_param_from_glm('residuals', labels, results, this_X, time_series=True)
                    r2[this_vox_idx] += get_param_from_glm('r_square', labels, results, this_X, time_series=False)

                    # Compute single-trial parameter (coded as "X")
                    cvec = np.zeros(this_X.shape[1])
                    cvec[this_X.columns.tolist().index('X')] = 1
                    con = compute_contrast(labels, results, con_val=cvec, contrast_type='t')
                    patterns[trial_nr, this_vox_idx] = getattr(con, STATS[cfg['pattern_units']])()

                    # Compute other conditions                    
                    for i, col in enumerate(cond_names):
                        cvec = np.zeros(this_X.shape[1])
                        cvec[this_X.columns.tolist().index(col)] = 1
                        con = compute_contrast(labels, results, con_val=cvec, contrast_type='t')
                        patterns[len(st_names)+i, this_vox_idx] += getattr(con, STATS[cfg['pattern_units']])()

                    # Evaluate "custom contrast" if there is any
                    if cfg['contrast'] is not None:
                        cvec = expression_to_contrast_vector(cfg['contrast'], this_X.columns.tolist())
                        con = compute_contrast(labels, results, con_val=cvec, contrast_type='t')
                        ccon[this_vox_idx] += getattr(con, STATS[cfg['pattern_units']])()

            # Because we estimated the betas for the other conditions len(st_names) times,
            # average them by dividing them by the number of trials
            patterns[len(st_names):, nonzero] /= st_names.size
            residuals[:, nonzero] /= st_names.size
            r2[nonzero] /= st_names.size

            # Also for the "single-trial stimulus intercept"
            st_idx_x = np.zeros(patterns.shape[0], dtype=bool)  
            st_idx_x[:len(st_names)] = True
            st_icept = patterns[:len(st_names), :].mean(axis=0)

            # ... and the custom contrast
            if cfg['contrast'] is not None:
                custom_contrast /= st_names.size
    
    # Extract single-trial (st) patterns and 
    # condition-average patterns (cond)
    stype = cfg['pattern_units']
    if cfg['single_trial_id'] is not None:
        st_patterns = patterns[st_idx_x, :]
        save_data(st_patterns, cfg, ddict, par_dir='best', run=run+1, desc='trial', dtype=stype, nii=True)
        cond_patterns = patterns[~st_idx_x, :]
    else:
        cond_patterns = patterns

    # Only save single-trial stimulus intercept if there are single trials
    #if cfg['single_trial_id'] is not None:
    #    save_data(st_icept, cfg, ddict, par_dir='best', run=run+1, desc='stimicept', dtype=stype, nii=True)

    # Save each parameter/statistic of the other conditions
    for i, name in enumerate(cond_names):    
        save_data(cond_patterns[i, :], cfg, ddict, par_dir='best', run=run+1, desc=name, dtype=stype, nii=True)

    # Always save residuals
    save_data(residuals, cfg, ddict, par_dir='best', run=run+1, desc='model', dtype='residuals', nii=True)

    # In case of LSA, also save predicted values
    if cfg['single_trial_model'] == 'lsa':
        save_data(preds, cfg, ddict, par_dir='best', run=run+1, desc='model', dtype='predicted', nii=True)    
        save_data(preds_icept, cfg, ddict, par_dir='best', run=run+1, desc='model', dtype='predictedicept', nii=True)
        tmp = np.ones_like(patterns)
        tmp *= patterns[-1, :]
        save_data(tmp, cfg, ddict, par_dir='best', run=run+1, desc='model', dtype='predictedunmod', nii=True)

    # Always save R2
    save_data(r2, cfg, ddict, par_dir='best', run=run+1, desc='model', dtype='r2', nii=True)

    # Save custom contrast (--contrast)
    if cfg['contrast'] is not None:
        save_data(custom_contrast, cfg, ddict, par_dir='best', run=run+1, desc='customcontrast', dtype=stype, nii=True)


def _run_glmdenoise_model(ddict, cfg, logger):
    """ Runs a GLMdenoise-style cross-validated analysis. """
    Y_all = ddict['denoised_func'].copy()
    nonzero = ~np.all(np.isclose(Y_all, 0.), axis=0)

    # Some shortcuts
    n_runs = np.unique(ddict['run_idx']).size
    K = Y_all.shape[1]
    stype = STATS[cfg['pattern_units']]

    # Pre-allocate some stuff, separately for bootstrap data (boot) and 
    # parameteric data (param)
    conditions = ddict['preproc_events']['trial_type'].unique().tolist()
    cond_boot = np.zeros((cfg['bootstraps'], len(conditions), K))
    cond_param = np.zeros((len(conditions), K))

    if cfg['contrast'] is not None:
        # ccon = custom contrast
        ccon_boot = np.zeros((cfg['bootstraps'], K))
        ccon_param = np.zeros(K)  # parametric

    # Note: opt_n_comps must be the same for each run!
    if ddict['opt_n_comps'].ndim > 1:
        raise ValueError("Cannot have run-specific n-comps when using GLMdenoise. Set --regularize-n-comps!")

    opt_n_comps = ddict['opt_n_comps']

    if cfg['hrf_model'] == 'kay':  # use optimal HRF
        if cfg['noiseproc_type'] == 'between':
            opt_hrf_idx = ddict['opt_hrf_idx'].astype(int)
        else:  # still have to do this!
            r2_hrf = _optimize_hrf_between(ddict, cfg, logger)
            opt_hrf_idx = r2_hrf.argmax(axis=0)
            save_data(opt_hrf_idx, cfg, ddict, par_dir='best', run=None, desc='opt', dtype='hrf')
            save_data(r2_hrf, cfg, ddict, par_dir='best', run=None, desc='hrf', dtype='r2')
            save_data(r2_hrf.max(axis=0), cfg, ddict, par_dir='best', run=None, desc='max', dtype='r2')   
    else:  # use the same HRF (this is ignored)
        opt_hrf_idx = np.zeros(K)

    # bootstrap indices [[1, 2, 3, 4], [1, 2, 2, 3], [3, 2, 1, 1]] etc.
    boots = [np.random.choice(np.arange(n_runs), size=n_runs)
             for _ in range(cfg['bootstraps'])]

    r2 = np.zeros(K)

    # Loop over HRF indices
    for hrf_idx in np.unique(opt_hrf_idx).astype(int):            
        # Loop over n-components
        for n_comp in np.unique(opt_n_comps):

            # Determine voxel index (intersection nonzero and the voxels that 
            # were denoised with the current n_comp)
            vox_idx = opt_n_comps == n_comp
            vox_idx = np.logical_and(vox_idx, nonzero)
            vox_idx = np.logical_and(vox_idx, hrf_idx == opt_hrf_idx)

            # Gather the run-specific design matrices
            Xs = []
            for run in range(n_runs):
                tr = ddict['trs'][run]
                this_Y, confs, events = get_run_data(ddict, run, func_type='denoised')
                ft = get_frame_times(tr, ddict, cfg, this_Y)
                # Note: hrf_idx is ignored when hrf_model is not "kay"
                X = create_design_matrix(tr, ft, events, hrf_model=cfg['hrf_model'], hrf_idx=hrf_idx)
                X = X.iloc[:, :-1]  # remove intercept

                # Orthogonalize noise components w.r.t. design matrix
                if n_comp != 0:
                    X.loc[:, :], _ = custom_clean(X, this_Y, confs[:, :n_comp], tr, ddict, cfg, clean_Y=False)
    
                X = X - X.mean(axis=0)
                Xs.append(X)

            # Concatenate design matrices
            X = pd.concat(Xs, axis=0)
            Y = Y_all[:, vox_idx]  # only current voxels

            # Get regular (parametric) scores
            labels, results = run_glm(Y, X.to_numpy(), noise_model='ols')
            r2[vox_idx] = get_param_from_glm('r_square', labels, results, X, time_series=False)

            for i, cond in enumerate(conditions):
                cvec = np.zeros(len(conditions))
                cvec[X.columns.tolist().index(cond)] = 1
                con = compute_contrast(labels, results, cvec)
                cond_param[i, vox_idx] = getattr(con, stype)()
            
            cvec = expression_to_contrast_vector(cfg['contrast'], X.columns.tolist())
            con = compute_contrast(labels, results, cvec)
            ccon_param[vox_idx] = getattr(con, stype)()
            """
            # Also get bootstrap scores
            for i_boot in tqdm(range(cfg['bootstraps'])):
                Xb, Yb = [], []
                for run_idx in boots[i_boot]:
                    Yb.append(Y[ddict['run_idx'] == run_idx, :])
                    Xb.append(X.loc[ddict['run_idx'] == run_idx, :])

                Xb = pd.concat(Xb, axis=0)
                Yb = np.vstack(Yb)

                labels, results = run_glm(Yb, Xb.to_numpy(), noise_model='ols')
                for i, cond in enumerate(conditions):
                    cvec = np.zeros(len(conditions))
                    cvec[X.columns.tolist().index(cond)] = 1
                    con = compute_contrast(labels, results, cvec)
                    cond_boot[i_boot, i, vox_idx] = getattr(con, stype)()
                
                cvec = expression_to_contrast_vector(cfg['contrast'], X.columns.tolist())
                con = compute_contrast(labels, results, cvec)
                ccon_boot[i_boot, vox_idx] = getattr(con, stype)()
            """

    save_data(r2, cfg, ddict, par_dir='best', run=None, desc='model', dtype='r2')
    # save stuff! Both bootstrap and parametric scores
    #cond_boot_mean = np.mean(cond_boot, axis=0)
    if cfg['pattern_units'] == 'beta':
        for i, cond in enumerate(conditions):
            #save_data(cond_boot_mean[i, :], cfg, ddict, par_dir='best', run=None, desc=cond, dtype='bootstrapbeta')
            save_data(cond_param[i, :], cfg, ddict, par_dir='best', run=None, desc=cond, dtype='parametricbeta')
    else:
        for i, cond in enumerate(conditions):
            #cond_boot_var = np.var(cond_boot, axis=0, ddof=1)
            #cond_t = cond_boot_mean / np.sqrt(cond_boot_var)
            #save_data(cond_t[i, :], cfg, ddict, par_dir='best', run=None, desc=cond, dtype='bootstraptstat')
            save_data(cond_param[i, :], cfg, ddict, par_dir='best', run=None, desc=cond, dtype='parametrictstat')

    if cfg['contrast'] is not None:
        #ccon_mean = np.mean(ccon_boot, axis=0)
        #ccon_var = np.var(ccon_boot, axis=0, ddof=1)
        #ccon_t = ccon_mean / np.sqrt(ccon_var)
        #save_data(ccon_mean, cfg, ddict, par_dir='best', run=None, desc='custom', dtype='bootstrapbeta')
        #save_data(ccon_t, cfg, ddict, par_dir='best', run=None, desc='custom', dtype='bootstraptstat')
        save_data(ccon_param, cfg, ddict, par_dir='best', run=None, desc='custom', dtype=f"parametric{cfg['pattern_units']}")


def _optimize_hrf_between(ddict, cfg, logger):
    """ Tries out 20 different Kay HRFs for a given run. """

    Y_all = ddict['denoised_func'].copy()
    nonzero = ~np.all(np.isclose(Y_all, 0.), axis=0)

    # Some shortcuts
    n_runs = np.unique(ddict['run_idx']).size
    K = Y_all.shape[1]

    # Pre-allocate R2 array: 20 (hrfs) x K (voxels)
    r2 = np.zeros((20, K))
    
    opt_n_comps = ddict['opt_n_comps']
    model = LinearRegression(fit_intercept=False, n_jobs=1)
    cv = LeaveOneGroupOut()

    # Loop over HRF indices
    for hrf_idx in tqdm_ctm(range(20), tdesc('Optimizing HRF shape: ')):           
        # Loop over n-components
        for n_comp in np.unique(opt_n_comps):

            # Determine voxel index (intersection nonzero and the voxels that 
            # were denoised with the current n_comp)
            vox_idx = opt_n_comps == n_comp
            vox_idx = np.logical_and(vox_idx, nonzero)

            # Gather the run-specific design matrices
            Xs = []
            for run in range(n_runs):
                tr = ddict['trs'][run]
                this_Y, confs, events = get_run_data(ddict, run, func_type='denoised')
                ft = get_frame_times(tr, ddict, cfg, this_Y)
                # Note: hrf_idx is ignored when hrf_model is not "kay"
                X = create_design_matrix(tr, ft, events, hrf_model=cfg['hrf_model'], hrf_idx=hrf_idx)
                X = X.iloc[:, :-1]  # remove intercept

                # Orthogonalize noise components w.r.t. design matrix
                if n_comp != 0:
                    X.loc[:, :], _ = custom_clean(X, this_Y, confs[:, :n_comp], tr, ddict, cfg, clean_Y=False)
    
                X = X - X.mean(axis=0)
                Xs.append(X)

            # Concatenate design matrices
            X = pd.concat(Xs, axis=0)
            Y = Y_all[:, vox_idx]  # only current voxels

            # Cross-validation across runs
            r2[hrf_idx, vox_idx] = cross_val_r2(model, X.to_numpy(), Y, cv=cv, groups=ddict['run_idx'])
    
    return r2
