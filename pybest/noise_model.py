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
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from .constants import ALPHAS
from .models import cross_val_r2

# IDEAS
# - "smarter" way to determine optimal alpha/n_comps (better than argmax); regularize
# - keep track of "optimal" predictors in _fit_ridge, so we don't have to refit
#   the model to regress it out?

def _run_parallel(run, ddict, cfg, logger, alphas, n_comps, cv):
    
    # Find indices of timepoints belong to this run
    t_idx = ddict['run_idx'] == run
    func = ddict['preproc_func'][t_idx, :]
    conf = ddict['preproc_conf'].loc[t_idx, :].to_numpy()
    K = func.shape[1]  # nr of voxels

    # Pre-allocate R2-scores (components x alphas x voxels)
    r2s = np.zeros((n_comps.size, alphas.size, K))

    # Loop over number of components
    for i, n_comp in enumerate(tqdm(n_comps, desc=f'run {run+1}')):
        # Check number of components
        if n_comp > conf.shape[1]:
            raise ValueError(f"Cannot select {n_comp} variables from conf data with {conf.shape[1]} components.")

        # Extract design matrix (with n_comp components)
        X = conf[:, :n_comp]

        # Loop across different regularization params
        # Note to self: we can use FastRidge here (pre-compute SVD)
        for ii, alpha in enumerate(alphas):
            # Get average predictions (across cv-repeats)
            model = Ridge(alpha=alpha, fit_intercept=False)
            r2s[i, ii, :] = cross_val_r2(model, X, func, cv)

    # Set voxels without signal to 0 (otherwise it'll have an R2 of 1)
    no_sig = func.mean(axis=0) == 0
    r2s[:, :, no_sig] = 0

    return r2s


def run_noise_processing(ddict, cfg, logger):
    """ Runs noise processing. """

    logger.info(f"Starting denoising with {cfg['n_comps']} components")
    n_comps = np.arange(1, cfg['n_comps']+1)  # range of components to test
    
    # Maybe add a "meta-seed" to cli options to ensure reproducibility?
    seed = np.random.randint(10e5)
    cv = RepeatedKFold(n_splits=cfg['cv_splits'], n_repeats=cfg['cv_repeats'], random_state=seed)
 
    #ddict['preproc_conf'].loc[:, :] = np.random.normal(0, 1, size=ddict['preproc_conf'].shape)
    
    # Parallel computation of R2 array (n_comps x alphas x voxels) across runs
    r2s_lst = Parallel(n_jobs=cfg['n_cpus'])(delayed(_run_parallel)(
        run, ddict, cfg, logger, ALPHAS, n_comps, cv)
        for run in np.unique(ddict['run_idx']).astype(int)
    )

    # Compute "optimal" parameters and save to disk for inspection
    sub, ses, task = cfg['sub'], cfg['ses'], cfg['task']
    ddict['opt_noise_alpha'] = []
    ddict['opt_noise_n_comps'] = []
    for run, r2s in enumerate(tqdm(r2s_lst)):
        K = r2s.shape[2]  # number of voxels
        # Compute maximum r2 across n-comps/alphas
        r2s_2d = r2s.reshape((np.prod(r2s.shape[:2]), K))
        r2_max = r2s_2d.max(axis=0)
        
        # Neat trick to do an argmax over two dims
        # opt_param_idx: 2 (ncomps, alpha) x K (vox)
        opt_param_idx = np.c_[np.unravel_index(
            r2s_2d.argmax(axis=0), shape=r2s.shape[:2]
        )].T.astype(int)

        # Set n_comps to -1 when R2 is negative (those voxels should not be denoised) 
        opt_param_idx[0, r2_max < 0] = -1
        
        # Extract *actual* optimal parameters (not their *indices*)
        # and mask voxels R2 < 0 in opt_n_comps
        opt_n_comps = n_comps[opt_param_idx[0, :]]
        opt_n_comps[r2_max < 0] = 0
        opt_alpha = ALPHAS[opt_param_idx[1, :]]
        
        # Find max r2 per n-comp (for inspection)
        r2_max_per_ncomp = np.zeros((n_comps.size, r2s.shape[2]))
        for i in range(n_comps.size):
            r2_max_per_ncomp[i, :] = r2s[i, :, :].max(axis=0)

        # Extract n_comps x alpha array (timepoints are n_comps, values are alpha)
        alpha_opt_per_ncomp = np.zeros((n_comps.size, r2s.shape[2]))
        for i in range(n_comps.size):
            alpha_opt_per_ncomp[i, :] = ALPHAS[r2s[i, :, :].argmax(axis=0)]

        # Save stuff
        out_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'denoising')
        if not op.isdir(out_dir):
            os.makedirs(out_dir)

        f_base = f'sub-{sub}_ses-{ses}_task-{task}_run-{run+1}_desc-'
        to_save = [  # This should always be saved
            (r2_max, 'max_r2'),
            (opt_alpha, 'opt_alpha'),
            (opt_n_comps, 'opt_ncomps'),
        ]    

        for dat, name in to_save:
            np.save(op.join(out_dir, f_base + name + '.npy'), dat)
            img = masking.unmask(dat, ddict['mask'])
            img.to_filename(op.join(out_dir, f_base + name + '.nii.gz'))
        
        if cfg['save_all']:
            img = masking.unmask(r2_max_per_ncomp, ddict['mask'])
            img.to_filename(op.join(out_dir, f_base + 'ncomp_r2.nii.gz'))
            img = masking.unmask(alpha_opt_per_ncomp, ddict['mask'])
            img.to_filename(op.join(out_dir, f_base + 'ncomp_alpha.nii.gz'))

        # Save for later
        ddict['opt_noise_alpha'].append(opt_alpha)
        ddict['opt_noise_n_comps'].append(opt_n_comps)

    return ddict


def load_denoised_data(ddict, cfg):
    
    sub, ses, task = cfg['sub'], cfg['ses'], cfg['task']
    preproc_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'preproc')
    denoising_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'denoising')

    ddict['opt_noise_alpha'] = np.vstack([np.load(f) for f in sorted(glob(op.join(denoising_dir, '*-opt_alpha.npy')))])
    ddict['opt_noise_n_comps'] = np.vstack([np.load(f) for f in sorted(glob(op.join(denoising_dir, '*-opt_ncomps.npy')))])
    
    ddict['preproc_conf'] = np.load(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_bold.npy'))
    ddict['preproc_conf'] = pd.read_csv(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_conf.tsv'), sep='\t')
    ddict['preproc_events'] = pd.read_csv(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_events.tsv'), sep='\t')
    ddict['mask'] = nib.load(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_mask.nii.gz'))
    ddict['run_idx'] = np.load(op.join(preproc_dir, 'run_idx.npy'))

    return ddict

    