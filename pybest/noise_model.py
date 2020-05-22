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
from sklearn.model_selection import RepeatedKFold

from .logging import tqdm_ctm, tdesc
from .models import cross_val_r2
from .utils import get_run_data

# IDEAS
# - "smarter" way to determine optimal n_comps (better than argmax)? regularize

def _run_parallel(run, ddict, cfg, logger, n_comps, cv):
    """ Function to run each run in parallel. """

    # Find indices of timepoints belong to this run
    func, conf, _ = get_run_data(ddict, run, func_type='preproc')
    nonzero = func.sum(axis=0) != 0  # exclude voxels w/o signal

    # Pre-allocate R2-scores (components x voxels)
    r2s = np.zeros((n_comps.size, func.shape[1]))

    # Loop over number of components
    model = LinearRegression(fit_intercept=False)
    for i, n_comp in enumerate(tqdm_ctm(n_comps, tdesc(f'Noise proc run {run+1}:'))):
        # Check number of components
        if n_comp > conf.shape[1]:
            raise ValueError(f"Cannot select {n_comp} variables from conf data with {conf.shape[1]} components.")

        # Extract design matrix (with n_comp components)
        X = conf[:, :n_comp]
        r2s[i, nonzero] = cross_val_r2(model, X, func[:, nonzero], cv)

    return r2s


def run_noise_processing(ddict, cfg, logger):
    """ Runs noise processing. """

    logger.info(f"Starting denoising with {cfg['n_comps']} components")
    n_comps = np.arange(1, cfg['n_comps']+1)  # range of components to test
    
    # Maybe add a "meta-seed" to cli options to ensure reproducibility?
    seed = np.random.randint(10e5)
    cv = RepeatedKFold(n_splits=cfg['cv_splits'], n_repeats=cfg['cv_repeats'], random_state=seed)

    # Uncomment line below to check how "well" it does with random confounds (= bias)
    #ddict['preproc_conf'].loc[:, :] = np.random.normal(0, 1, size=ddict['preproc_conf'].shape)
    
    # Parallel computation of R2 array (n_comps x voxels) across runs
    r2s_list = Parallel(n_jobs=cfg['n_cpus'])(delayed(_run_parallel)(
        run, ddict, cfg, logger, n_comps, cv)
        for run in np.unique(ddict['run_idx'])
    )

    sub, ses, task = cfg['sub'], cfg['ses'], cfg['task']

    # Pre-allocate parameter array (runs x voxels)
    # Note to self: make sure these are ints! Otherwise yield_unique_params doesn't work
    ddict['opt_noise_n_comps'] = np.zeros((len(r2s_list), ddict['preproc_func'].shape[1]), dtype=int)

    # Pre-allocate clean func
    func_clean = ddict['preproc_func'].copy()
    for run, r2s in enumerate(tqdm_ctm(r2s_list, tdesc('Denoising funcs: '))):
        
        # Compute maximum r2 across n-comps
        r2_max = r2s.max(axis=0)
        opt_n_comps_idx = r2s.argmax(axis=0)
        opt_n_comps = n_comps[opt_n_comps_idx.astype(int)]
        opt_n_comps[r2_max < 0] = 0  # set negative r2 voxels to 0 comps
    
        # Save (but maybe not necessary, given that denoising is done
        # in this step, not in the signal model)
        ddict['opt_noise_n_comps'][run, :] = opt_n_comps

        # Start denoising! Loop over unique indices
        func, conf, _ = get_run_data(ddict, run, func_type='preproc')
        nonzero = func.sum(axis=0) != 0
        model = LinearRegression(fit_intercept=False)
        for this_n_comps in np.unique(opt_n_comps):
            # If n_comps is 0, then R2 was negative and we
            # don't want to denoise, so continue
            if this_n_comps == 0:
                continue

            # Find voxels that correspond to this_n_comps
            vox_idx = opt_n_comps == this_n_comps
            # Exclude voxels without signal
            vox_idx = np.logical_and(vox_idx, nonzero)

            X = conf[:, :this_n_comps]
            # Refit model on all data this time and remove fitted values
            func[:, vox_idx] -= model.fit(X, func[:, vox_idx]).predict(X)

        # Standardize once more
        func = signal.clean(func, detrend=False, standardize='zscore')
        func_clean[ddict['run_idx'] == run, :] = func

        # Save stuff
        out_dir = op.join(cfg['save_dir'], 'denoising')
        if not op.isdir(out_dir):
            os.makedirs(out_dir)

        f_base = f'sub-{sub}_ses-{ses}_task-{task}_run-{run+1}_desc-'
        to_save = [  # This should always be saved
            (r2_max, 'max_r2'),
            (opt_n_comps, 'opt_ncomps'),
            (func, 'denoised_bold')
        ]    

        for dat, name in to_save:
            np.save(op.join(out_dir, f_base + name + '.npy'), dat)
            if ddict['mask'] is not None:
                img = masking.unmask(dat, ddict['mask'])
                img.to_filename(op.join(out_dir, f_base + name + '.nii.gz'))
        
        if cfg['save_all']:
            img = masking.unmask(r2s, ddict['mask'])
            img.to_filename(op.join(out_dir, f_base + 'ncomp_r2.nii.gz'))

    f_out = op.join(out_dir, cfg['f_base'] + '_desc-denoised_bold.npy')
    np.save(f_out, func_clean)

    ddict['denoised_func'] = func_clean
    ddict['opt_noise_n_comps'] = np.vstack(ddict['opt_noise_n_comps']).astype(int)

    return ddict


def load_denoising_data(ddict, cfg):
    """ Loads the denoising parameters/data. """

    sub, ses, task = cfg['sub'], cfg['ses'], cfg['task']
    preproc_dir = op.join(cfg['save_dir'], 'preproc')
    denoising_dir = op.join(cfg['save_dir'], 'denoising')

    ddict['opt_noise_n_comps'] = np.vstack([np.load(f) for f in sorted(glob(op.join(denoising_dir, '*-opt_ncomps.npy')))]).astype(int)
    
    ddict['denoised_func'] = np.load(op.join(denoising_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-denoised_bold.npy'))
    ddict['preproc_conf'] = pd.read_csv(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_conf.tsv'), sep='\t')
    ddict['preproc_events'] = pd.read_csv(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_events.tsv'), sep='\t')
    
    if 'fs' in cfg['space']:
        ddict['mask'] = None
    else:
        ddict['mask'] = nib.load(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_mask.nii.gz'))
    ddict['run_idx'] = np.load(op.join(preproc_dir, 'run_idx.npy'))

    return ddict

    