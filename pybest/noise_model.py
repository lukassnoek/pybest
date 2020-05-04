import os
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from nilearn import masking
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def _run_parallel(run, ddict, cfg, logger, n_repeats, alphas, n_comps):
    
    # Find indices of timepoints belong to this run
    idx = ddict['run_idx'] == run
    func = ddict['preproc_func'][idx, :]
    conf = ddict['preproc_conf'].loc[idx, :].to_numpy()
    K = func.shape[1]  # nr of voxels
        
    # Pre-allocate R2-scores (components x alphas x voxels)
    r2s = np.zeros((n_comps.size, ALPHAS.size, K))

    # Loop over number of components
    for i, n_comp in enumerate(tqdm(n_comps, desc=f'run {run+1}')):
        # Extract design matrix (with n_comp components)
        X = conf[:, :(n_comp+1)]            

        # Loop across different regularization params
        # Note to self: we can use FastRidge here
        for ii, alpha in enumerate(ALPHAS):
            # Pre-allocate prediction array
            preds = np.zeros_like(func)

            # Use repeated KFold for stability (averaged over later)
            for _ in range(N_REPEATS):
                cv = KFold(n_splits=5, shuffle=True)
                model = Ridge(alpha=alpha, fit_intercept=False)

                # Start cross-validation loop
                for train_idx, test_idx in cv.split(X):
                    model.fit(X[train_idx, :], func[train_idx, :])
                    preds[test_idx, :] += model.predict(X[test_idx, :])
            
            # Average predictions across repeats and compute R2
            preds /= N_REPEATS
            r2s[i, ii, :] = r2_score(func, preds, multioutput='raw_values')

    return r2s


def run_noise_processing(ddict, cfg, logger):
    """ Runs noise processing.

    Parameters
    ----------
    func_data : np.ndarray
        2D (time x voxels) preprocessed fMRI data
    conf_data : pd.DataFrame
        Dataframe (time x pca-decomposed confounds)
    run_idx : np.ndarray
        1D (time) run index
    """
    
    # hard-coded params
    N_REPEATS = 2
    ALPHAS = np.array([0, 0.01, 1, 10, 100, 500, 1000, 5000, 10000])
    
    n_comps = np.arange(0, cfg['ncomps'])  # range of components to test

    logger.info(f"Starting denoising with {cfg['ncomps']} components")
    r2s_lst = Parallel(n_jobs=cfg['nthreads'])(delayed(_run_parallel)(
        run, ddict, cfg, logger, N_REPEATS, ALPHAS, N_COMPS) 
        for run in np.unique(ddict['run_idx']).astype(int)
    )

    for i, r2s in enumerate(r2s_lst):
        # Compute max R2 and associated "optimal" hyperparameters,
        # n-components and alpha 
        r2s_2D = r2s.reshape((np.prod(r2s.shape[:2]), K))
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
        n_comps_range = np.zeros((n_comps.size, K))
        for i in range(n_comps.size):
            n_comps_range[i, :] = r2s[i, :, :].max(axis=0)

        out_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'denoising')
        if not op.isdir(out_dir):
            os.makedirs(out_dir)

        f_base = f'sub-{sub}_ses-{ses}_task-{task}'
        f_out = op.join(out_dir, f_base + '_desc-preproc_bold.npy')
        np.save(f_out, ddict['preproc_func'])
    
        masking.unmask(n_comps_range, ddict['mask']).to_filename(f'r2_comps_run{run}.nii.gz')
        masking.unmask(max_r2, ddict['mask']).to_filename(f'opt_r2_run{run}.nii.gz')
        masking.unmask(opt_n_comps, ddict['mask']).to_filename(f'opt_n_comps_run{run}.nii.gz')
        masking.unmask(opt_alpha, ddict['mask']).to_filename(f'opt_alpha_run{run}.nii.gz')


def save_denoised_data(sub, ses, task, ddict, cfg):
    
    out_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'denoised')
    if not op.isdir(out_dir):
        os.makedirs(out_dir)

    f_base = f'sub-{sub}_ses-{ses}_task-{task}'
    f_out = op.join(out_dir, f_base + '_desc-denoised_bold.npy')
    np.save(f_out, ddict['dns_func'])

    func_data_img = masking.unmask(ddict['dns_func'], ddict['mask'])
    func_data_img.to_filename(f_out.replace('npy', 'nii.gz'))


def load_denoised_data(sub, ses, task, ddict, cfg):
    
    preproc_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'preproc')
    denoised_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'denoised')

    ddict['dns_func'] = np.load(op.join(denoised_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-denoised_bold.npy'))
    ddict['preproc_events'] = pd.read_csv(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_events.tsv'), sep='\t')
    ddict['mask'] = nib.load(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_mask.nii.gz'))
    ddict['run_idx'] = np.load(op.join(preproc_dir, 'run_idx.npy'))

    return ddict

    