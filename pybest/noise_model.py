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
from sklearn.model_selection import KFold


def _run_parallel(run, ddict, cfg, logger, alphas, n_comps, seeds):
    
    # Find indices of timepoints belong to this run
    t_idx = ddict['run_idx'] == run
    func = ddict['preproc_func'][t_idx, :]
    conf = ddict['preproc_conf'].loc[t_idx, :].to_numpy()
    K = func.shape[1]  # nr of voxels
        
    # Pre-allocate R2-scores (components x alphas x voxels)
    r2s = np.zeros((n_comps.size, alphas.size, K))

    # Loop over number of components
    for i, n_comp in enumerate(tqdm(n_comps, desc=f'run {run+1}')):
        # Extract design matrix (with n_comp components)
        if n_comp > conf.shape[1]:
            raise ValueError(f"Cannot select {n_comp} variables from conf data with {conf.shape[1]} components.")

        X = conf[:, :n_comp]
        # Loop across different regularization params
        # Note to self: we can use FastRidge here
        for ii, alpha in enumerate(alphas):
            # Pre-allocate prediction array
            preds = np.zeros_like(func)

            # Use repeated KFold for stability (averaged over later)
            # Extract function from this (also used in main func)
            for iii in range(cfg['cv_repeats']):
                cv = KFold(n_splits=cfg['cv_splits'], shuffle=True, random_state=seeds[iii])
                model = Ridge(alpha=alpha, fit_intercept=False)

                # Start cross-validation loop
                for train_idx, test_idx in cv.split(X):
                    model.fit(X[train_idx, :], func[train_idx, :])
                    preds[test_idx, :] += model.predict(X[test_idx, :])
            
            # Average predictions across repeats and compute R2
            preds /= cfg['cv_repeats']
            r2s[i, ii, :] = r2_score(func, preds, multioutput='raw_values')

    # Set voxels without signal to 0 (otherwise it'll be 1)
    no_sig = func.mean(axis=0) == 0
    r2s[:, :, no_sig] = 0

    return r2s


def run_noise_processing(ddict, cfg, logger):
    """ Runs noise processing. """

    logger.info(f"Starting denoising with {cfg['ncomps']} components")
    
    # ALPHAS is so far hard-coded
    ALPHAS = np.array([0, 0.01, 1, 10, 100, 500, 1000, 5000])    
    n_comps = np.arange(1, cfg['ncomps']+1)  # range of components to test
    
    # Maybe add a "meta-seed" to cli options to ensure reproducibility?
    seeds = np.random.randint(low=0, high=100000, size=cfg['cv_repeats'])

    r2s_lst = Parallel(n_jobs=cfg['nthreads'])(delayed(_run_parallel)(
        run, ddict, cfg, logger, ALPHAS, n_comps, seeds)
        for run in np.unique(ddict['run_idx']).astype(int)
    )
    sub, ses, task = cfg['sub'], cfg['ses'], cfg['task']
    denoised_func = np.zeros_like(ddict['preproc_func'])
    for run, r2s in enumerate(tqdm(r2s_lst)):
        # Compute max R2 and associated "optimal" hyperparameters,
        # n-components and alpha: ncomps x alphas x voxels
        r2s_2D = r2s.reshape((np.prod(r2s.shape[:2]), r2s.shape[2]))
        max_r2 = r2s_2D.max(axis=0)  # best possible r2
        
        # Neat trick to do an argmax over two dims
        # opt_params: 2 (ncomps, alpha) x K (vox)
        opt_params = np.c_[np.unravel_index(
            r2s_2D.argmax(axis=0), shape=r2s.shape[:2]
        )].T.astype(int)

        # Set "bad voxels" n_comps parameter to 0        
        opt_params[0, max_r2 < 0] = 0

        # Remove noise (only to check)
        t_idx = ddict['run_idx'] == run
        func = ddict['preproc_func'][t_idx, :]
        conf = ddict['preproc_conf'].loc[t_idx, :].to_numpy()
       
        # this_denoised_func (corresponds to current run)
        this_denoised_func = np.zeros_like(func)

        # uniq_combs: unique combinations of optimal parameters (2 x combs)
        uniq_combs = np.unique(opt_params, axis=1).astype(int)
        for uix in range(uniq_combs.shape[1]):  # loop over combinations
            these_params = uniq_combs[:, uix]

            # Which voxels have this combination of optimal params?
            vox_idx = np.all(opt_params == these_params[:, np.newaxis], axis=0)

            # Which parameters (n_comp, alpha) belong to this combination?
            n_comp = n_comps[these_params[0]]
            alpha = ALPHAS[these_params[1]]
            if n_comp == 0:  # do not denoise when R2 < 0
                this_denoised_func[:, vox_idx] = to_denoise
                continue

            # Index func data / confound matrix
            to_denoise = func[:, vox_idx]
            X = conf[:, :n_comp]
            
            preds = np.zeros_like(to_denoise)
            model = Ridge(alpha=alpha, fit_intercept=False)

            # Use repeated KFold for stability (averaged over later)
            for iii in range(cfg['cv_repeats']):
                # Need to fix the shuffle (should be the same as earlier)
                cv = KFold(n_splits=cfg['cv_splits'], shuffle=True, random_state=seeds[iii])
                
                # Start cross-validation loop
                for train_idx, test_idx in cv.split(X):
                    model.fit(X[train_idx, :], to_denoise[train_idx, :])
                    preds[test_idx, :] += model.predict(X[test_idx, :])
        
            # Average predictions across repeats and subtract from func
            preds /= cfg['cv_repeats']
            this_denoised_func[:, vox_idx] = to_denoise - preds

        # Extract actual optimal parameters (not indices)
        opt_n_comps = n_comps[opt_params[0, :]]
        opt_alpha = ALPHAS[opt_params[1, :]]

        # Compute optimal R2 and mask voxels r2 < 0 in opt_comp
        opt_n_comps[max_r2 < 0] = 0

        # Extract component-wise max
        n_comps_range = np.zeros((n_comps.size, r2s.shape[2]))
        for i in range(n_comps.size):
            n_comps_range[i, :] = r2s[i, :, :].max(axis=0)

        out_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'denoising')
        if not op.isdir(out_dir):
            os.makedirs(out_dir)

        f_base = f'sub-{sub}_ses-{ses}_task-{task}_run-{run+1}_desc-'
        to_save = [
            (max_r2, 'max_r2'), (opt_alpha, 'opt_alpha'),
            (opt_n_comps, 'opt_ncomps'), (n_comps_range, 'ncomps_r2'),
            (this_denoised_func, 'denoised_bold')
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

    # Bit hacky (but good for RAM)
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

    