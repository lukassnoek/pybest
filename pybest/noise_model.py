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
from fracridge import FracRidge


# IDEAS
# - "smarter" way to determine optimal alpha/n_comps (better than argmax); regularize
# - keep track of "optimal" predictors in _fit_ridge, so we don't have to refit
#   the model to regress it out?

def _fit_ridge(X, y, cv, fracs):
    # Pre-allocate r2 array
    r2s = np.zeros((fracs.size, y.shape[-1]))
    alphas = np.zeros((fracs.size, y.shape[-1]))
    model = FracRidge(fracs=fracs, fit_intercept=False)
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx, :], y[train_idx, :])
        alphas += model.alpha_

        # preds: n-test x alphas x voxels
        preds = model.predict(X[test_idx, :])
        for ii in range(fracs.size):
            r2s[ii, :] += r2_score(y[test_idx], preds[:, ii, :], multioutput='raw_values') 

    # Average across folds and repeats
    r2s /= cv.get_n_splits()
    alphas /= cv.get_n_splits()
    return r2s, alphas


def _run_parallel(run, ddict, cfg, logger, n_comps, fracs, cv):
    
    # Find indices of timepoints belong to this run
    t_idx = ddict['run_idx'] == run
    func = ddict['preproc_func'][t_idx, :]
    conf = ddict['preproc_conf'].loc[t_idx, :].to_numpy()
    K = func.shape[1]  # nr of voxels

    # Voxels with actual signal
    with_sig = func.mean(axis=0) != 0
    
    # Pre-allocate R2-scores (components x alphas x voxels)
    r2s = np.zeros((n_comps.size, fracs.size, K))
    alphas = np.zeros((n_comps.size, fracs.size, K))
    # Loop over number of components
    for i, n_comp in enumerate(tqdm(n_comps, desc=f'run {run+1}')):
        # Check number of components
        if n_comp > conf.shape[1]:
            raise ValueError(f"Cannot select {n_comp} variables from conf data with {conf.shape[1]} components.")

        # Extract design matrix (with n_comp components)
        X = conf[:, :n_comp]

        # Loop across different regularization params
        # Note to self: we can use FastRidge here (pre-compute SVD)
        out = _fit_ridge(X, func[:, with_sig], cv, fracs)
        r2s[i, :, with_sig], alphas[i, :, with_sig] = out[0].T, out[1].T

    return r2s, alphas


def run_noise_processing(ddict, cfg, logger):
    """ Runs noise processing. """

    logger.info(f"Starting denoising with {cfg['n_comps']} components")
    n_comps = np.arange(2, cfg['n_comps']+1)  # range of components to test
    fracs = np.linspace(0, 1, cfg['n_alphas'], endpoint=True)

    # Maybe add a "meta-seed" to cli options to ensure reproducibility?
    cv = RepeatedKFold(
        n_splits=cfg['cv_splits'],
        n_repeats=cfg['cv_repeats'],
        random_state=np.random.randint(10e5)
    )

    #ddict['preproc_conf'].loc[:, :] = np.random.normal(0, 1, size=ddict['preproc_conf'].shape)
    # Parallel computation of R2 array (n_comps x alphas x voxels) across runs
    out = Parallel(n_jobs=cfg['nthreads'])(delayed(_run_parallel)(
        run, ddict, cfg, logger, n_comps, fracs, cv)
        for run in np.unique(ddict['run_idx']).astype(int)
    )

    # Compute "optimal" parameters and save to disk for inspection
    sub, ses, task = cfg['sub'], cfg['ses'], cfg['task']
    denoised_func = np.zeros_like(ddict['preproc_func'])
    for run, (r2s, alphas) in enumerate(tqdm(out)):
        
        # Compute max R2 and associated "optimal" hyperparameters,
        # n-components and alpha: ncomps x alphas x voxels
        r2s_2D = r2s.reshape((np.prod(r2s.shape[:2]), r2s.shape[2]))
        r2_max = r2s_2D.max(axis=0)  # best possible r2
        
        # Neat trick to do an argmax over two dims
        # opt_param_idx: 2 (ncomps, alpha) x K (vox)
        opt_param_idx = np.c_[np.unravel_index(
            r2s_2D.argmax(axis=0), shape=r2s.shape[:2]
        )].T.astype(int)

        # Extract data to be denoised
        t_idx = ddict['run_idx'] == run
        func = ddict['preproc_func'][t_idx, :]
        conf = ddict['preproc_conf'].loc[t_idx, :].to_numpy()

        with_sig = func.sum(axis=0) != 0

        # Set n_comps to -1 when R2 is negative (those voxels should not be denoised) 
        opt_param_idx[0, r2_max < 0] = -1
        opt_param_idx[0, ~with_sig] = -1

        # this_denoised_func (corresponds to current run)
        this_denoised_func = func.copy()

        # uniq_combs: unique combinations of optimal parameter indices (2 x combs)
        uniq_combs = np.unique(opt_param_idx, axis=1).astype(int)
        #max_r2_check = np.zeros(max_r2.size)
        """
        for uix in range(uniq_combs.shape[1]):  # loop over combinations
            # Current combination of n_comps and alpha *indices*
            these_param_idx = uniq_combs[:, uix]

            # Which voxels have this combination of optimal params?
            vox_idx = np.all(opt_param_idx == these_param_idx[:, np.newaxis], axis=0)
            vox_idx = np.logical_and(vox_idx, with_sig)

            # Which parameters (n_comp, alpha) belong to this combination?
            n_comp = n_comps[these_param_idx[0]]
            alpha = alphas[these_param_idx[0], these_param_idx[1]]
            if n_comp == -1:  # do not denoise when R2 < 0
                continue

            # Index func data / confound matrix
            to_denoise = func[:, vox_idx]
            X = conf[:, :n_comp]

            # Get predictions
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X, func)
            #preds = _fit_ridge(cfg['cv_repeats'], cfg['cv_splits'], alpha, seeds, X, to_denoise)
            this_denoised_func[:, vox_idx] = to_denoise - model.predict(X)
            #max_r2_check[vox_idx] = r2_score(to_denoise, preds, multioutput='raw_values')
            opt_alpha[vox_idx] = alpha
            opt_n_comps[vox_idx] = n_comp
        """
        # Extract actual optimal parameters (not indices)
        #opt_alpha = alphas[opt_param_idx[0, :], opt_param_idx[]]
        opt_frac = fracs[opt_param_idx[1, :]]
        opt_n_comps = n_comps[opt_param_idx[0, :]]

        # Compute optimal R2 and mask voxels R2 < 0 in opt_comp
        opt_n_comps[r2_max < 0] = 0

        # Extract component-wise max
        n_comps_range = np.zeros((n_comps.size, r2s.shape[2]))
        for i in range(n_comps.size):
            n_comps_range[i, :] = r2s[i, :, :].max(axis=0)

        # Extract n_comps x alpha array (timepoints are n_comps, values are alpha)
        n_comps_by_frac = np.zeros((n_comps.size, r2s.shape[2]))
        for i in range(n_comps.size):
            n_comps_by_frac[i, :] = fracs[r2s[i, :, :].argmax(axis=0)]

        # Save stuff
        out_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'denoising')
        if not op.isdir(out_dir):
            os.makedirs(out_dir)

        f_base = f'sub-{sub}_ses-{ses}_task-{task}_run-{run+1}_desc-'
        to_save = [
            (r2_max, 'max_r2'), (opt_frac, 'opt_frac'), #(opt_alpha, 'opt_alpha'), 
            (opt_n_comps, 'opt_ncomps'), (n_comps_range, 'ncomps_r2'),
            (n_comps_by_frac, 'ncomps_frac'), (this_denoised_func, 'denoised_bold')
        ]
        for dat, name in to_save:
            np.save(op.join(out_dir, f_base + name + '.npy'), dat)
            img = masking.unmask(dat, ddict['mask'])
            img.to_filename(op.join(out_dir, f_base + name + '.nii.gz'))

        # Save run into concat time series
        denoised_func[t_idx, :] = this_denoised_func
        
    f_out = f'sub-{sub}_ses-{ses}_task-{task}_desc-denoised_bold.nii.gz'
    img = masking.unmask(denoised_func, ddict['mask'])
    img.to_filename(op.join(out_dir, f_out))

    # Get 4D files with parameters: X x Y x Z x (params)
    alpha_files = sorted(glob(op.join(out_dir, '*desc-opt_alpha.nii.gz')))
    ncomps_files = sorted(glob(op.join(out_dir, '*desc-opt_ncomps.nii.gz')))
    ddict['alpha_data'] = image.concat_imgs(alpha_files)
    ddict['ncomps_data'] = image.concat_imgs(ncomps_files)
    return ddict


def load_denoised_data(ddict, cfg):
    
    sub, ses, task = cfg['sub'], cfg['ses'], cfg['task']
    preproc_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'preproc')
    denoising_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'denoising')

    ddict['alpha_data'] = image.concat_imgs(sorted(glob(op.join(denoising_dir, '*desc-opt_alpha.nii.gz'))))
    ddict['ncomps_data'] = image.concat_imgs(sorted(glob(op.join(denoising_dir, '*desc-opt_ncomps.nii.gz'))))
    ddict['preproc_conf'] = np.load(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_bold.npy'))
    ddict['preproc_conf'] = pd.read_csv(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_conf.tsv'), sep='\t')
    ddict['preproc_events'] = pd.read_csv(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_events.tsv'), sep='\t')
    ddict['mask'] = nib.load(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_mask.nii.gz'))
    ddict['run_idx'] = np.load(op.join(preproc_dir, 'run_idx.npy'))

    return ddict

    