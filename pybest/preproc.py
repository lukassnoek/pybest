import os
import os.path as op
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from nilearn import image, masking, signal
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from nistats.design_matrix import _cosine_drift as dct_set
from sklearn.decomposition import PCA, FastICA
from .utils import _load_gifti


def _run_func_parallel(ddict, cfg, run, func, logger):
    
    # Load data
    if 'fs' in cfg['space']:  # assume gifti
        data = _load_gifti(func)
    else:
        # Mask data and extract stuff
        data = masking.apply_mask(func, ddict['mask'])

    # By now, data is a 2D array (time x voxels)
    data = hp_filter(data, ddict, cfg, logger)
    
    # Add to run index
    run_idx = np.ones(data.shape[0]) * run
    return data, run_idx


def preprocess_funcs(ddict, cfg, logger):
    """ Preprocesses a set of functional files (either volumetric nifti or
    surface gifti); masking, high-pass filter (DCT) and normalization only.
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

    out = Parallel(n_jobs=cfg['n_cpus'])(delayed(_run_func_parallel)
        (ddict, cfg, run, func, logger)
        for run, func in enumerate(tqdm(ddict['funcs']))
    )

    logger.info("Saving preprocessed data to disk")
    out_dir = op.join(cfg['out_dir'], 'preproc')
    if not op.isdir(out_dir):
        os.makedirs(out_dir)

    if cfg['save_all']:  # Save run-wise data as niftis for inspection
        for i, (data, _) in enumerate(out):
            # maybe other name/desc (is the same as fmriprep output now)
            f_out = op.join(out_dir, cfg['f_base'] + f'_run-{i+1}_desc-preproc_bold.nii.gz')
            masking.unmask(data, ddict['mask']).to_filename(f_out)

    # Concatenate data in time dimension
    data = np.vstack([d[0] for d in out])
    run_idx = np.concatenate([r[1] for r in out]).astype(int)

    f_out = op.join(out_dir, cfg['f_base'] + '_desc-preproc_bold.npy')
    np.save(f_out, data)
    np.save(op.join(out_dir, 'run_idx.npy'), run_idx)

    # TO FIX: write out all data if there's no mask
    f_out = f_out.replace('bold.npy', 'mask.nii.gz')
    if ddict['mask'] is not None:
        ddict['mask'].to_filename(f_out)

    logger.info(
        f"HP-filtered/normalized func data has {data.shape[0]} timepoints "
        f"(across {i+1} runs) and {data.shape[1]} voxels"
    )

    # Store in data-dictionary (ddict)
    ddict['preproc_func'] = data
    ddict['run_idx'] = run_idx
    return ddict


def preprocess_confs(ddict, cfg, logger):
    """ Preprocesses confounds by doing the following:
    1. Horizontal concatenation of Fmriprep confounds and RETROICOR (if any)
    2. Set NaNs to 0
    3. High-pass the data (same as functional data)
    4. PCA
    """
    
    logger.info("Starting preprocessing of confounds")
    decomp = PCA() if cfg['decomp'] == 'pca' else FastICA(max_iter=1000)
    
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
        
        # Perform PCA
        data = decomp.fit_transform(data)
        if data.shape[1] < cfg['n_comps']:
            cfg['n_comps'] = data.shape[1]
            logger.warning(
                f"Setting n-comps to {cfg['n_comps']}, because the {cfg['decomp']} "
                 "decomposition yielded fewer components than n-comps."
            )

        # Extract desired number of components
        data = data[:, :cfg['n_comps']]

        # Make proper dataframe
        cols = [f'decomp_{str(c+1).zfill(3)}' for c in range(data.shape[1])]
        data = pd.DataFrame(data, columns=cols)
        data_.append(data)

    out_dir = op.join(cfg['out_dir'], 'preproc')
    if cfg['save_all']:
        for i, data in enumerate(data_):
            f_out = op.join(out_dir, cfg['f_base'] + f'_run-{i+1}_desc-preproc_conf.tsv')
            data.to_csv(f_out, sep='\t', index=False)

    # Concatenate DataFrames and save
    data = pd.concat(data_, axis=0)
    f_out = op.join(out_dir, cfg['f_base'] + '_desc-preproc_conf.tsv')
    data.to_csv(f_out, sep='\t', index=False)    

    logger.info(
        f"HP-filtered/normalized confound data has {data.shape[0]} timepoints "
        f"(across {i+1} runs) and {data.shape[1]} components."
    )
    ddict['preproc_conf'] = data
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

    out_dir = op.join(cfg['out_dir'], 'preproc')
    if cfg['save_all']:
        for i, data in enumerate(data_):
            f_out = op.join(out_dir, cfg['f_base'] + f'_run-{i+1}_desc-preproc_events.tsv')
            data.to_csv(f_out, sep='\t', index=False)

    # Adjust onsets for concatenated events file
    for run in np.unique(ddict['run_idx']).astype(int):
        prev_run = ddict['run_idx'] < run
        data_[run].loc[:, 'onset'] = data_[run].loc[:, 'onset'] + prev_run.sum() * ddict['tr']

    data = pd.concat(data_, axis=0)
    f_out = op.join(out_dir, cfg['f_base'] + '_desc-preproc_events.tsv')
    data.to_csv(f_out, sep='\t', index=False)
    
    logger.info(f"Found in total {data.shape[0]} events across {i+1} runs")
    ddict['preproc_events'] = data
    return ddict


def hp_filter(data, ddict, cfg, logger):
    """ High-pass filter (DCT or Savitsky-Golay). """
    n_vol = data.shape[0]
    tr = ddict['tr']
    st_ref = cfg['slice_time_ref']
    frame_times = np.linspace(st_ref * tr, n_vol * (tr + st_ref), n_vol, endpoint=False)

    # Create high-pass filter and clean
    if cfg['high_pass_type'] == 'dct':
        hp_set = dct_set(cfg['high_pass'], frame_times)
        data = signal.clean(data, detrend=False, standardize='zscore', confounds=hp_set)
    else:  # savgol, hardcode polyorder
        window = int(np.round((1 / cfg['high_pass']) / tr))
        hp_sig = savgol_filter(data, window_length=window, polyorder=2, axis=0)
        data -= hp_sig
        data = signal.clean(data, detrend=False, standardize='zscore')

    return data


def load_preproc_data(ddict, cfg):
    """ Loads preprocessed data. """
    sub, ses, task = cfg['sub'], cfg['ses'], cfg['task']
    in_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'preproc')
    f_base = f'sub-{sub}_ses-{ses}_task-{task}_desc-preproc_'
    
    f_in = op.join(in_dir, f_base)
    ddict['preproc_func'] = np.load(f_in + 'bold.npy')
    ddict['preproc_conf'] = pd.read_csv(f_in + 'conf.tsv', sep='\t')
    ddict['preproc_events'] = pd.read_csv(f_in + 'events.tsv', sep='\t')
    ddict['mask'] = None if 'fs' in cfg['space'] else nib.load(f_in + 'mask.nii.gz')
    ddict['run_idx'] = np.load(op.join(in_dir, 'run_idx.npy'))

    return ddict