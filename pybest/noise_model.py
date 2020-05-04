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


def _run_parallel(run, ddict, cfg, logger, alphas, n_comps):
    
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
        X = conf[:, :(n_comp+1)]            

        # Loop across different regularization params
        # Note to self: we can use FastRidge here
        for ii, alpha in enumerate(alphas):
            # Pre-allocate prediction array
            preds = np.zeros_like(func)

            # Use repeated KFold for stability (averaged over later)
            for _ in range(cfg['cv_repeats']):
                cv = KFold(n_splits=cfg['cv_splits'], shuffle=True)
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
    
    # hard-coded params
    ALPHAS = np.array([0, 0.01, 1, 10, 100, 500, 1000, 5000, 10000])    
    n_comps = np.arange(0, cfg['ncomps'])  # range of components to test

    logger.info(f"Starting denoising with {cfg['ncomps']} components")
    r2s_lst = Parallel(n_jobs=cfg['nthreads'])(delayed(_run_parallel)(
        run, ddict, cfg, logger, ALPHAS, n_comps) 
        for run in np.unique(ddict['run_idx']).astype(int)
    )
    sub, ses, task = cfg['sub'], cfg['ses'], cfg['task']
    denoised_func = np.zeros_like(ddict['preproc_func'])
    for run, r2s in enumerate(tqdm(r2s_lst)):
        # Compute max R2 and associated "optimal" hyperparameters,
        # n-components and alpha 
        r2s_2D = r2s.reshape((np.prod(r2s.shape[:2]), r2s.shape[2]))
        max_r2 = r2s_2D.max(axis=0)
        
        # Neat trick to do an argmax over two dims
        opt_n_comps_idx, opt_alpha_idx = np.unravel_index(
            r2s_2D.argmax(axis=0), shape=r2s.shape[:2]
        )

        # Use index to extract actual values
        opt_n_comps = n_comps[opt_n_comps_idx] + 1
        opt_alpha = ALPHAS[opt_alpha_idx]

        # Compute optimal R2 and mask voxels < 0 in opt_comp
        bad_vox = np.logical_and(opt_n_comps == 0, max_r2 < 0)
        opt_n_comps[bad_vox] = 0

        # Extract component-wise max
        n_comps_range = np.zeros((n_comps.size, r2s.shape[2]))
        for i in range(n_comps.size):
            n_comps_range[i, :] = r2s[i, :, :].max(axis=0)

        # Remove noise (only to check)
        t_idx = ddict['run_idx'] == run
        func = ddict['preproc_func'][t_idx, :]
        conf = ddict['preproc_conf'].loc[t_idx, :].to_numpy()
       
        this_denoised_func = np.zeros_like(func)
        for n_comp in np.unique(opt_n_comps):
            comp_idx = opt_n_comps == n_comp

            for alpha in np.unique(opt_alpha):
                alpha_idx = opt_alpha == alpha
                vox_idx = np.logical_and(comp_idx, alpha_idx)
                print(vox_idx.sum())
                to_denoise = func[:, vox_idx]
                X = conf[:, :n_comp]  # not plus 1, we did this earlier
                
                preds = np.zeros_like(to_denoise)
                model = Ridge(alpha=alpha, fit_intercept=False)

                # Use repeated KFold for stability (averaged over later)
                for _ in range(cfg['cv_repeats']):
                    # Need to fix the shuffle (should be the same as earlier)
                    cv = KFold(n_splits=cfg['cv_splits'], shuffle=True)
                    
                    # Start cross-validation loop
                    for train_idx, test_idx in cv.split(X):
                        model.fit(X[train_idx, :], to_denoise[train_idx, :])
                        preds[test_idx, :] += model.predict(X[test_idx, :])
            
                # Average predictions across repeats and compute R2
                preds /= cfg['cv_repeats']
                this_denoised_func[:, vox_idx] = to_denoise - preds

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

        # Save run into concat time seriess
        denoised_func[t_idx, :] = this_denoised_func
        
    f_out = f'sub-{sub}_ses-{ses}_task-{task}_desc-denoised_bold.nii.gz'
    img = masking.unmask(denoised_func, ddict['mask'])
    img.to_filename(op.join(out_dir, f_out))

    # Bit hacky (but good for RAM)
    ddict['alpha_data'] = image.concat_imgs(sorted(glob(out_dir, '*desc-opt_alpha.nii.gz')))
    ddict['ncomps_data'] = image.concat_imgs(sorted(glob(out_dir, '*desc-opt_ncomps.nii.gz')))
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

    