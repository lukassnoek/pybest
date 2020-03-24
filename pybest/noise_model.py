import os
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from nilearn import masking
from .utils import tqdm_out


def run_noise_processing(func_data, conf_data, run_idx, denoising_strategy, logger):
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

    logger.info(f"Starting denoising using strategy '{denoising_strategy}'")

    if denoising_strategy == 'dummy':
        return func_data

    """ OLD STUFF
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


def save_denoised_data(sub, ses, task, func_data, mask, work_dir):
    
    out_dir = op.join(work_dir, f'sub-{sub}', f'ses-{ses}', 'denoised')
    if not op.isdir(out_dir):
        os.makedirs(out_dir)

    f_base = f'sub-{sub}_ses-{ses}_task-{task}'
    f_out = op.join(out_dir, f_base + '_desc-denoised_bold.npy')
    np.save(f_out, func_data)

    func_data_img = masking.unmask(func_data, mask)
    func_data_img.to_filename(f_out.replace('npy', 'nii.gz'))


def load_denoised_data(sub, ses, task, work_dir):
    
    preproc_dir = op.join(work_dir, f'sub-{sub}', f'ses-{ses}', 'preproc')
    denoised_dir = op.join(work_dir, f'sub-{sub}', f'ses-{ses}', 'denoised')

    func_data = np.load(op.join(denoised_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-denoised_bold.npy'))
    event_data = pd.read_csv(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_events.tsv'))
    mask = nib.load(op.join(preproc_dir, f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_mask.nii.gz'))
    run_idx = np.load(op.join(preproc_dir, 'run_idx.npy'))

    return func_data, event_data, mask, run_idx

    