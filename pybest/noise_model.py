import os
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, make_scorer
from nilearn import masking
from .utils import tqdm_out


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
    denoising_strategy : str
        Method for denoising (for now: only dummy; does nothing)
    """
    strategy = cfg['denoising_strategy']
    logger.info(f"Starting denoising using strategy '{strategy}'")

    #if strategy == 'dummy':
    #    ddict['dns_func'] = ddict['preproc_func']
    #    return ddict
    cv = KFold(n_splits=5)
    for run in np.unique(ddict['run_idx']).astype(int):
        idx = ddict['run_idx'] == run
        func = ddict['preproc_func'][idx, :]
        conf = ddict['preproc_conf'].loc[idx, :].to_numpy()
        
        # Try out comp 1-20
        n_comps = np.arange(0, 100, 2)
        r2s = np.zeros((n_comps.size, func.shape[1]))
        for i, n_comp in enumerate(tqdm(n_comps, desc=f'run {run+1}')):
            X = conf[:, :(n_comp+1)]
            preds = cross_val_predict(RidgeCV(), X, func, cv=cv)
            r2s[i, :] = r2_score(func, preds, multioutput='raw_values')

        opt_comp = r2s.argmax(axis=0)
        opt_r2 = r2s.max(axis=0)
        bad_vox = np.logical_and(opt_comp == 0, opt_r2 < 0)
        opt_comp[bad_vox] = 0
        
        clean_func = np.zeros_like(func)
        for comp in np.unique(opt_comp):
            vox_idx = opt_comp == comp
            # TO DO: REFIT MODEL WITH APPROPRIATE LAMBDA,
            # REGRESS OUT STUFF
            func[:, vox_idx] = ...
       
        #masking.unmask(r2s, ddict['mask']).to_filename(f'r2_run{run}.nii.gz')

            
    """
    scaler = StandardScaler()
    model = RidgeCV()

    n_run = np.unique(run_idx).size
    cv = GroupKFold(n_splits=n_run).split(conf_data, func_data[:, 0], groups=run_idx)
    
    r2_scores = np.zeros(func_data.shape[1])
    for train_idx, test_idx in tqdm(cv, file=tqdm_out):
        
        y_train = scaler.fit_transform(func_data[train_idx, :])
        y_test = scaler.fit_transform(func_data[test_idx, :])

        X_train = scaler.fit_transform(conf_data.iloc[train_idx, :])
        X_test = scaler.fit_transform(conf_data.iloc[test_idx, :])

        model.fit(X_train, y_train)
        # Overfitting to check
        y_pred = model.predict(X_train)
        r2_scores += r2_score(y_train, y_pred, multioutput='raw_values')
        
    r2_scores /= n_run
    if mask is not None:
        r2_scores = masking.unmask(r2_scores, mask)
        r2_scores.to_filename(op.join(work_dir, 'r2.nii.gz'))
    else:
        np.save('r2.npy', r2_scores)
    """


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

    