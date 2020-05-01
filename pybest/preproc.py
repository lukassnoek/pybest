import os
import os.path as op
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from nilearn import image, masking, signal
from scipy.signal import savgol_filter
from nistats.design_matrix import _cosine_drift as dct_set
from sklearn.decomposition import PCA
from .utils import _load_gifti, tqdm_out


def preprocess_funcs(ddict, cfg, logger):
    """ Preprocesses a set of functional files (either volumetric nifti or
    surface gifti); high-pass filter (DCT) and normalization only.

    Parameters
    ----------
    funcs : list
        List of paths to functional files
    mask : str
        Path to mask (for now: assume it's a GM probseg file)
    space : str
        Name of space ('T1w', 'fsaverage{5,6}', 'MNI152NLin2009cAsym')
    logger : logging object
        Main logger (from cli.py)
    high_pass_type : str
        Either 'dct' or 'savgol'
    high_pass : float
        High-pass cutoff (in Hz)
    savgol_order : int
        Savitsky-Golay polyorder (default: 4)
    gm_thresh : float
        Gray matter probability threshold (higher = included in binary mask)
    tr : float
        TR of scan (only relevant if space is fsaverage) in seconds

    Returns
    -------
    data_ : np.ndarray
        2D array (time x voxels) with run-wise concatenated data
    run_idx_ : np.ndarray
        1D array with run indices, [0, 0, 0, ... , 1, 1, 1, ... , R, R, R]
        for R runs
    """

    if 'fs' not in cfg['space']:  # no need for mask in surface files
        if ddict['gm_prob'] is None:  # use functional brain masks
            logger.info("Creating mask by intersection of function masks")
            fmasks = [f.replace('desc-preproc_bold', 'desc-brain_mask') for f in ddict['funcs']]
            mask = masking.intersect_masks(fmasks, threshold=0.8)
        else:
            # Using provided masks
            logger.info("Creating mask using GM probability map")

            # Downsample (necessary by default)
            gm_prob = image.resample_to_img(ddict['gm_prob'], ddict['funcs'][0])
            gm_prob_data = gm_prob.get_fdata()

            # Threshold
            gm_prob_data = (gm_prob_data >= cfg['gm_thresh']).astype(int)
            mask = nib.Nifti1Image(gm_prob_data, affine=gm_prob.affine)
    else:
        # If fsaverage{5,6} space, don't use any mask
        mask = None

    ddict['mask'] = mask
    logger.info("Starting preprocessing of functional data ... ")

    data_, run_idx_ = [], []
    for i, func in enumerate(tqdm(ddict['funcs'], file=tqdm_out)):

        # Load data
        if 'fs' in cfg['space']:  # assume gifti
            data = _load_gifti(func)
        else:
            # Mask data and extract stuff
            data = masking.apply_mask(func, ddict['mask'])

        # By now, data is a 2D array (time x voxels)
        data = hp_filter(data, ddict, cfg, logger)
        data_.append(data)

        # Add to run index
        run_idx_.append(np.ones(data.shape[0]) * i)

    # Concatenate data in time dimension (or should we keep it in lists?)
    data_ = np.vstack(data_)
    run_idx_ = np.concatenate(run_idx_)

    logger.info(
        f"HP-filtered/normalized func data has {data_.shape[0]} timepoints "
        f"(across {i+1} runs) and {data_.shape[1]} voxels"
    )
    ddict['preproc_func'] = data_
    ddict['run_idx'] = run_idx_
    return ddict


def preprocess_confs(ddict, cfg, logger):
    """ Preprocesses confounds by doing the following:
    1. Horizontal concatenation of Fmriprep confounds and RETROICOR (if any)
    2. Set NaNs to 0
    3. High-pass the data (same as functional data)
    4. PCA

    Parameters
    ----------

    Returns
    -------

    """
    
    logger.info("Starting preprocessing of confounds")
    pca = PCA()
    
    data_ = []
    for i, conf in enumerate(ddict['confs']):

        # Load and remove cosine regressors
        data = pd.read_csv(conf, sep='\t')
        to_remove = [col for col in data.columns if 'cosine' in col]
        data = data.drop(to_remove, axis=1)

        if ddict['ricors'] is not None:  # add RETROICOR regressors, if any
            ricor_data = pd.read_csv(ddict['ricors'][i], sep='\t')
            data = pd.concat((data, ricor_data), axis=1)
        
        logger.info(f"Loaded {data.shape[1]} confound variables for run {i+1}")
        data = data.fillna(0)

        # High-pass confounds
        data = hp_filter(data.to_numpy(), ddict, cfg, logger)
        
        # Perform PCA and store in dataframe
        # Only store 100 components (might change)
        data = pca.fit_transform(data)[:, :100]
        n_comp_for_90r2 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9)
        logger.info(f"PCA needed {n_comp_for_90r2} components to explain 90% of the variance for run {i+1}")

        cols = [f'pca_{str(c+1).zfill(3)}' for c in range(data.shape[1])]
        data = pd.DataFrame(data, columns=cols)
        data_.append(data)

    # data_ is a 
    data_ = pd.concat(data_, axis=0)#, join='inner', ignore_index=True)
    logger.info(
        f"HP-filtered/normalized confound data has {data_.shape[0]} timepoints "
        f"(across {i+1} runs) and {data_.shape[1]} components."
    )
    ddict['preproc_conf'] = data_
    return ddict


def preprocess_events(ddict, cfg, logger):
    """ Preprocesses event files. """
    to_keep = ['onset', 'duration', 'trial_type']
    data_ = []
    for i, event in enumerate(ddict['events']):
        data = pd.read_csv(event, sep='\t')
        for col in to_keep:
            if col not in data.columns:
                raise ValueError(f"No column '{col}' in {event}!")

        data = data.loc[:, to_keep]
        n_uniq = data['trial_type'].unique().size
        perc_unique = int(np.round((n_uniq / data.shape[0]) * 100))
        logger.info(f"Found {n_uniq} unique trial types for run {i+1} ({perc_unique} %)")
        data['run'] = i+1
        data_.append(data)

    data_ = pd.concat(data_, axis=0)
    logger.info(f"Found in total {data_.shape[0]} events across {i+1} runs")
    ddict['preproc_events'] = data_
    return ddict


def hp_filter(data, ddict, cfg, logger):
    n_vol = data.shape[0]
    tr = ddict['tr']
    frame_times = np.linspace(0.5 * tr, n_vol * (tr + 0.5), n_vol, endpoint=False)

    # Create high-pass filter and clean
    if cfg['high_pass_type'] == 'dct':
        hp_set = dct_set(cfg['high_pass'], frame_times)[:, :-1]  # remove intercept
        data = signal.clean(data, detrend=False, standardize='zscore', confounds=hp_set)
    else:  # savgol, hardcode polyorder
        window = int(np.round((1 / cfg['high_pass']) / tr))
        hp_sig = savgol_filter(data, window_length=window, polyorder=3, axis=0)
        data -= hp_sig
        data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    return data


def save_preproc_data(sub, ses, task, ddict, cfg):

    out_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'preproc')
    if not op.isdir(out_dir):
        os.makedirs(out_dir)

    f_base = f'sub-{sub}_ses-{ses}_task-{task}'
    f_out = op.join(out_dir, f_base + '_desc-preproc_bold.npy')
    np.save(f_out, ddict['preproc_func'])
    
    # TO FIX: write out all data if there's no mask
    func_data_img = masking.unmask(ddict['preproc_func'], ddict['mask'])
    func_data_img.to_filename(f_out.replace('npy', 'nii.gz'))

    np.save(op.join(out_dir, 'run_idx.npy'), ddict['run_idx'])

    f_out = f_out.replace('bold.npy', 'mask.nii.gz')
    if ddict['mask'] is not None:
        ddict['mask'].to_filename(f_out)

    f_out = f_out.replace('mask.nii.gz', 'conf.tsv')
    ddict['preproc_conf'].to_csv(f_out, sep='\t', index=False)

    f_out = f_out.replace('conf', 'events')
    ddict['preproc_events'].to_csv(f_out, sep='\t', index=False)


def load_preproc_data(sub, ses, task, ddict, cfg):
    
    in_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'preproc')
    f_base = f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_'
    
    f_in = op.join(in_dir, f_base)
    ddict['preproc_func'] = np.load(f_in + 'bold.npy')
    ddict['preproc_conf'] = pd.read_csv(f_in + 'conf.tsv', sep='\t')
    ddict['preproc_events'] = pd.read_csv(f_in + 'events.tsv', sep='\t')
    ddict['mask'] = nib.load(f_in + 'mask.nii.gz')
    ddict['run_idx'] = np.load(op.join(in_dir, 'run_idx.npy'))

    return ddict