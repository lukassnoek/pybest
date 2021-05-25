import os
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from glob import glob
from nilearn import masking, signal, image
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut

from .logging import tqdm_ctm, tdesc
from .utils import get_run_data, get_frame_times, create_design_matrix, hp_filter
from .utils import save_data, load_gifti, custom_clean, argmax_regularized, load_and_split_cifti
from .models import cross_val_r2


def run_noise_processing(ddict, cfg, logger):
    """ Runs noise processing either within runs (i.e., separately for each run)
    when signalproc-type == 'single-trial' or across runs (i.e., on the run-concatenated data)
    using a cross-validated analysis when signalproc-type == 'glmdenoise'. """

    K = ddict['preproc_func'].shape[1]
    n_runs = len(ddict['trs'])

    if cfg['skip_noiseproc']:
        logger.warn("Skipping noise processing (because of --skip-noiseproc)")
        
        # Pretend that "preprocessed" data is "denoised" data
        ddict['denoised_func'] = ddict['preproc_func']
        # Mock opt_n_comps
        if cfg['signalproc_type'] == 'glmdenoise':
            ddict['opt_n_comps'] = np.zeros(K)
        else:
            ddict['opt_n_comps'] = np.zeros((n_runs, K))
    
        # save mock opt_n_comps
        save_data(ddict['opt_n_comps'], cfg, ddict, par_dir='denoising', run=None,
                  desc='opt', dtype='ncomps', nii=True)

        return ddict

    logger.info(f"Starting denoising with {cfg['n_comps']} components")

    # Within-run (cross-validated) confound regression
    if cfg['noiseproc_type'] == 'within':
        # Must be > 0
        n_comps = np.arange(1, cfg['n_comps']+1).astype(int)  # range of components to test

        # Denoising is done within runs!
        # Maybe add a "meta-seed" to cli options to ensure reproducibility?
        seed = np.random.randint(10e5)
        cv = RepeatedKFold(n_splits=cfg['cv_splits'], n_repeats=cfg['cv_repeats'], random_state=seed)

        # Parallel computation of R2 array (n_comps x voxels) across runs (list has len(runs))
        r2s_list = Parallel(n_jobs=cfg['n_cpus'])(delayed(_run_parallel_within_run)(
            run, ddict, cfg, logger, n_comps, cv) for run in range(n_runs)
        )

        # r2: runs x n_comps x voxels
        r2 = np.stack(r2s_list)

        # Regularize = same opt_n_comps for each run
        if cfg['regularize_n_comps']:

            # Compute median across runs
            r2_median = np.median(r2, axis=0)  # median across runs
            save_data(r2_median, cfg, ddict, par_dir='denoising', run=None,
                      nii=True, desc='ncomps', dtype='medianr2')

            # Maximum r2 across n-comps
            r2_max = r2_median.max(axis=0)

            # Use de-meaned median score to selected cutoff
            r2_median_tmp = np.median(r2 - r2.mean(axis=0), axis=0)  # median across runs

            opt_n_comps_idx = argmax_regularized(r2_median_tmp, axis=0, percent=cfg['argmax_percent'])
            opt_n_comps = n_comps[opt_n_comps_idx.astype(int)]
            opt_n_comps[r2_max < 0] = 0
            ddict['opt_n_comps'] = opt_n_comps
            
            for run in range(n_runs):
                r2_ncomps = r2[run, :, :]
                r2_max = r2_ncomps.max(axis=0)
                save_data(r2_max, cfg, ddict, par_dir='denoising', run=run+1, nii=True, desc='max', dtype='r2')
                save_data(r2_ncomps, cfg, ddict, par_dir='denoising', run=run+1, nii=True, desc='ncomps', dtype='r2')
        else:
            # Per-run optimal n-comps
            ddict['opt_n_comps'] = n_comps[argmax_regularized(r2, axis=1, percent=cfg['argmax_percent'])]
            
            # Determine, per runs, the optimal number of noise comps
            for run in range(n_runs):
                r2_ncomps = r2[run, :, :]
                # Compute maximum r2 across n-comps
                r2_max = r2_ncomps.max(axis=0)

                # Whenever r2 < 0, set opt_n_comps to zero (no denoising)
                ddict['opt_n_comps'][run, r2_max < 0] = 0

                if cfg['save_all']:  # save per-run statistics
                    opt_n_comps = ddict['opt_n_comps'][run, :]
                    to_save = [(r2_ncomps, 'ncomps', 'r2'), (r2_max, 'max', 'r2'), (opt_n_comps, 'opt', 'ncomps')]
                    for data, desc, dtype in to_save:
                        save_data(data, cfg, ddict, par_dir='denoising', run=run+1, desc=desc, dtype=dtype, nii=True)

    else:  # between-run, GLMdenoise style denoising
        # Also check 0 components!
        n_comps = np.arange(0, cfg['n_comps']+1).astype(int)  # range of components to test
    
        # Initialize R2 array (across HRFs/n-components/voxels)
        cv = LeaveOneGroupOut()
        r2s_list = Parallel(n_jobs=cfg['n_cpus'])(delayed(_run_parallel_across_runs)(
            ddict, cfg, logger, this_n_comp, cv) for this_n_comp
            in tqdm_ctm(n_comps, tdesc(f'Noise proc: '))
        )
        
        # r2: hrfs x n_components x voxels
        r2 = np.moveaxis(np.stack(r2s_list), [0, 1], [1, 0])
        
        # Best score across HRFs
        r2_ncomps = r2.max(axis=0)

        # Best overall r2 (across HRFs and n_comps)
        r2_max = r2_ncomps.max(axis=0)

        # Find optimal number of components and HRF index
        opt_n_comps = n_comps[argmax_regularized(r2_ncomps, axis=0, percent=cfg['argmax_percent'])]
        opt_n_comps[r2_max < 0] = 0
        opt_hrf_idx = np.zeros(K)

        for i in n_comps:
            idx = opt_n_comps == i
            # Do not regularize HRF index
            opt_hrf_idx[idx] = r2[:, i, idx].argmax(axis=0)

        # Always save the following:
        save_data(opt_hrf_idx, cfg, ddict, par_dir='denoising', run=None, desc='opt', dtype='hrf', nii=True)
        save_data(opt_n_comps, cfg, ddict, par_dir='denoising', run=None, desc='opt', dtype='ncomps', nii=True)

        if cfg['save_all']:
            to_save = [(r2_ncomps, 'ncomps', 'r2'), (r2_ncomps.max(axis=0), 'max', 'r2')]
            for data, desc, dtype in to_save:
                save_data(data, cfg, ddict, par_dir='denoising', run=None, desc=desc, dtype=dtype, nii=True)

        ddict['opt_hrf_idx'] = opt_hrf_idx
        ddict['opt_n_comps'] = opt_n_comps

    ### START DENOISING PROCESS ###

    # Pre-allocate clean func
    func_clean = ddict['preproc_func'].copy()
    for run in tqdm_ctm(range(n_runs), tdesc('Denoising funcs: ')):

        # If we have a run-specific optimal n-comps, use it
        if ddict['opt_n_comps'].ndim > 1:
            opt_n_comps = ddict['opt_n_comps'][run, :]
        else:  # otherwise use the "regularized" optimal n-comps (same for all runs)
            opt_n_comps = ddict['opt_n_comps']

        # Loop over unique indices
        func, conf, _ = get_run_data(ddict, run, func_type='preproc')
        nonzero = ~np.all(np.isclose(func, 0.), axis=0)  # mask
        for this_n_comps in np.unique(opt_n_comps).astype(int):
            # If n_comps is 0, then R2 was negative and we
            # don't want to denoise, so continue
            if this_n_comps == 0:
                continue

            # Find voxels that correspond to this_n_comps
            vox_idx = opt_n_comps == this_n_comps
            # Exclude voxels without signal
            vox_idx = np.logical_and(vox_idx, nonzero)

            C = conf[:, :this_n_comps]
            # Refit model on all data this time and remove fitted values
            func[:, vox_idx] = signal.clean(func[:, vox_idx], detrend=False, confounds=C, standardize=False)

        # Standardize once more
        func = signal.clean(func, detrend=False, standardize='zscore')
        func_clean[ddict['run_idx'] == run, :] = func

        # Save denoised data
        if cfg['save_all']:
            save_data(func, cfg, ddict, par_dir='denoising', run=run+1, desc='denoised',
                      dtype='bold', skip_if_single_run=True, nii=True)

    # Always save full denoised timeseries (and optimal number of components for each run)
    save_data(func_clean, cfg, ddict, par_dir='denoising', run=None, desc='denoised', dtype='bold', nii=False)
    save_data(ddict['opt_n_comps'], cfg, ddict, par_dir='denoising', run=None, desc='opt', dtype='ncomps', nii=True)

    ddict['denoised_func'] = func_clean
    return ddict


def _run_parallel_within_run(run, ddict, cfg, logger, n_comps, cv):
    """ Function to evaluate noise model parallel across runs.
    Only used when signalproc-type == 'single-trial', because in case of
    'glmdenoise', the noise model is evaluated across runs
    """

    # Find indices of timepoints belong to this run
    func, conf, _ = get_run_data(ddict, run, func_type='preproc')
    nonzero = ~np.all(np.isclose(func, 0.), axis=0)
    
    # Pre-allocate R2-scores (components x voxels)
    r2s = np.zeros((n_comps.size, func.shape[1]))

    # Loop over number of components
    model = LinearRegression(fit_intercept=False, n_jobs=1)
    for i, n_comp in enumerate(tqdm_ctm(n_comps, tdesc(f'Noise proc run {run+1}:'))):
        # Check number of components
        if n_comp > conf.shape[1]:
            raise ValueError(f"Cannot select {n_comp} variables from conf data with {conf.shape[1]} components.")

        # Extract design matrix (with n_comp components)
        C = conf[:, :n_comp]
        r2s[i, nonzero] = cross_val_r2(model, C, func[:, nonzero], cv)

    return r2s


def _run_parallel_across_runs(ddict, cfg, logger, this_n_comp, cv):
    """ Run, per HRF, a cv model. """
    K = ddict['preproc_func'].shape[1]
    n_runs = np.unique(ddict['run_idx']).size

    if cfg['hrf_model'] == 'kay':
        r2 = np.zeros((20, K))
        to_iter = range(20)
    else:
        r2 = np.zeros((1, K))
        to_iter = range(1)

    # Define model (linreg) and cross-validation routine (leave-one-run-out)        
    model = LinearRegression(fit_intercept=False, n_jobs=1)
    # Define fMRI data (Y) and full confound matrix (C)
    Y = ddict['preproc_func'].copy()
 
    # Loop over HRFs
    for i in to_iter:
        Xs = []  # store runwise design matrix
        # Create run-wise design matrix
        for run in range(n_runs):
            t_idx = ddict['run_idx'] == run
            this_Y, conf, events = get_run_data(ddict, run=run, func_type='preproc')
            tr = ddict['trs'][run]
            ft = get_frame_times(tr, ddict, cfg, this_Y)
            if this_n_comp == 0:
                C = None
            else:
                C = conf[:, :this_n_comp]  # extract first `this_n_comp` columns
            
            # create actual DM
            X = create_design_matrix(tr, ft, events, hrf_model=cfg['hrf_model'], hrf_idx=i)
            X = X.drop('constant', axis=1)

            # We can't cross-validate with a single-trial design, so we'll just the
            # "stimulus intercept" to cross-validate instead
            if cfg['single_trial_id'] is not None:
                st_idx = X.columns.str.contains(cfg['single_trial_id'])
                X['unmodstim'] = X.loc[:, st_idx].sum(axis=1)
                X = X.loc[:, ['unmodstim']]  # remove single trials

            # Filter and remove confounds (C) from both the design matrix (X) and data (Y)
            X.loc[:, :], this_Y = custom_clean(X, this_Y, C, tr, ddict, cfg, clean_Y=True)
            X = X - X.mean(axis=0)
            Xs.append(X)
            Y[t_idx, :] = this_Y

        # Concatenate across runs + standardize Y
        X = pd.concat(Xs, axis=0).to_numpy()
        Y = signal.clean(Y, detrend=False, standardize='zscore')

        # Cross-validation across runs
        r2[i, :] = cross_val_r2(model, X, Y, cv=cv, groups=ddict['run_idx'])

    return r2


def load_denoising_data(ddict, cfg):
    """ Loads the denoising parameters/data. Ugh, so ugly. Need to refactor sometime. """
    f_base = cfg['f_base']
    preproc_dir = op.join(cfg['save_dir'], 'preproc')
    denoising_dir = op.join(cfg['save_dir'], 'denoising')

    # Load in denoised data
    if cfg['skip_noiseproc']:  # load in preproc data if skipping noiseproc
        ddict['denoised_func'] = np.load(op.join(preproc_dir, f'{f_base}_desc-preproc_bold.npy'))
    else:
        ddict['denoised_func'] = np.load(op.join(denoising_dir, f'{f_base}_desc-denoised_bold.npy'))
    if 'fs' in cfg['space']:
        ddict['mask'] = None
    else:
        ddict['mask'] = nib.load(op.join(preproc_dir, f'{f_base}_desc-preproc_mask.nii.gz'))

    if 'fs' in cfg['space']:
        if cfg['iscift'] == 'y':
            ddict['trs'] = [load_and_split_cifti(f, cfg['atlas_file'], cfg['left_id'],
                                                 cfg['right_id'], cfg['subc_id'])[1] for f in ddict['funcs']]
        else:
            ddict['trs'] = [load_gifti(f)[1] for f in ddict['funcs']]
        ddict['opt_n_comps'] = np.load(op.join(denoising_dir, f'{f_base}_desc-opt_ncomps.npy'))
        if cfg['hrf_model'] == 'kay' and cfg['signalproc_type'] == 'glmdenoise':
            ddict['opt_hrf_idx'] = np.load(op.join(denoising_dir, f'{f_base}_desc-opt_hrf.npy'))
    else:
        ddict['trs'] = [nib.load(f).header['pixdim'][4] for f in ddict['funcs']]
        # For some reason, the line below takes a long time to run
        ddict['opt_n_comps'] = masking.apply_mask(op.join(denoising_dir, f'{f_base}_desc-opt_ncomps.nii.gz'), ddict['mask'])
        if cfg['hrf_model'] == 'kay' and cfg['signalproc_type'] == 'glmdenoise':
            ddict['opt_hrf_idx'] = masking.apply_mask(op.join(denoising_dir, f'{f_base}_desc-opt_hrf.nii.gz'), ddict['mask'])
        
    ddict['preproc_conf'] = pd.read_csv(op.join(preproc_dir, f'{f_base}_desc-preproc_conf.tsv'), sep='\t')

    if not cfg['skip_signalproc']:
        f_events = op.join(preproc_dir, f'{f_base}_desc-preproc_events.tsv')
        ddict['preproc_events'] = pd.read_csv(f_events, sep='\t')
    else:
        ddict['preproc_events'] = None

    ddict['run_idx'] = np.load(op.join(preproc_dir, f"task-{cfg['c_task']}_run_idx.npy"))
    
    return ddict
