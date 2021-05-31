import os
import os.path as op
import numpy as np
import nibabel as nib
import pandas as pd
import re
from tqdm import tqdm
from nilearn import image, masking, signal
from joblib import Parallel, delayed
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.linear_model import LinearRegression

from .logging import tqdm_ctm, tdesc
from .models import cross_val_r2
from .utils import load_gifti, get_frame_times, create_design_matrix, hp_filter, save_data, load_and_split_cifti


def preprocess_funcs(ddict, cfg, logger):
    """ Preprocesses a set of functional files (either volumetric nifti or
    surface gifti); masking, high-pass filter (DCT) and normalization only.
    """

    if 'fs' not in cfg['space']:  # no need for mask in surface files
        if ddict['gm_prob'] is None:  # use functional brain masks
            logger.info("Creating mask by intersection of fMRI masks")
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
        for run, func in enumerate(tqdm_ctm(ddict['funcs'], tdesc('Preprocessing funcs:')))
    )

    # Concatenate data in time dimension
    data = np.vstack([d[0] for d in out])
    run_idx = np.concatenate([r[1] for r in out]).astype(int)

    # Save functional data, ALWAYS as npy file (saves time/disk space)
    save_data(data, cfg, ddict, par_dir='preproc', run=None, desc='preproc', dtype='bold')

    # Save run_idx
    out_dir = op.join(cfg['save_dir'], 'preproc')
    np.save(op.join(out_dir, f"task-{cfg['c_task']}_run_idx.npy"), run_idx)

    # Extract TRs
    ddict['trs'] = [o[2] for o in out]
    logger.info(f"Found the following TRs across runs: {ddict['trs']}")
    
    # Save mask
    save_data(ddict['mask'], cfg, ddict, par_dir='preproc', run=None, desc='preproc', dtype='mask', nii=True)

    # Store in data-dictionary (ddict)
    ddict['preproc_func'] = data
    ddict['run_idx'] = run_idx
    return ddict


def _run_func_parallel(ddict, cfg, run, func, logger):
    """ Parallelizes loading/hp-filtering of fMRI run. """
    # Load data
    if 'fs' in cfg['space']:  # assume gifti
        if cfg['iscifti'] == 'y':
            data, tr = load_and_split_cifti(func, cfg['atlas_file'],cfg, cfg['left_id'], cfg['right_id'], cfg['subc_id'], cfg['mode'])
            tr /= 1000 # defined in msec
        else:
            data, tr = load_gifti(func, cfg, return_tr=True)
            tr /= 1000  # defined in msec
    else:
        # Load/mask data and extract stuff
        tr = nib.load(func).header['pixdim'][4]
        data = masking.apply_mask(func, ddict['mask'])

    # By now, data is a 2D array (time x voxels)
    data = hp_filter(data, tr, ddict, cfg)
    
    # Add to run index
    run_idx = np.ones(data.shape[0]) * run

    if cfg['save_all']:  # Save run-wise data as niftis for inspection
        save_data(data, cfg, ddict, par_dir='preproc', run=run+1, nii=True,
                  desc='preproc', dtype='bold', skip_if_single_run=True)

    return data, run_idx, tr


def preprocess_confs_fmriprep(ddict, cfg, logger):
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
        start_tr = [item[1] if re.search(item[0], conf, re.IGNORECASE) else 0 for item in cfg.get('skip_tr')][0]
        data = pd.read_csv(conf, sep='\t')[start_tr:]
        if cfg['confounds_filter'] is not None:
            confounds = cfg.get('confounds_filter')
            if type(confounds) == str:
                data = data.filter(regex=confounds)
            else:
                col_reg = re.compile('|'.join(confounds))
                data = data.filter(regex=col_reg)


        # Remove cosines and confounds related to the global signal
        # Anecdotal evidence that leaving out the global signal gives better results ...
        to_remove = [col for col in data.columns if 'cosine' in col or 'global' in col]
        data = data.drop(to_remove, axis=1)
        if cfg['ricor_dir'] is not None and ddict['ricors']:  # add RETROICOR regressors, if any
            ricor_data = pd.read_csv(ddict['ricors'][i], sep='\t')
            data = pd.concat((data, ricor_data), axis=1)
        
        logger.info(f"Loaded {data.shape[1]} confound variables for run {i+1}")
        data = data.fillna(0)

        # High-pass confounds
        data = hp_filter(data.to_numpy(), ddict['trs'][i], ddict, cfg)
        
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

        # Apply HP filter again (not sure if necessary), but a good idea
        # for proper orthogonalization
        data = hp_filter(data, ddict['trs'][i], ddict, cfg, standardize='zscore')

        # Make proper dataframe
        cols = [f'decomp_{str(c+1).zfill(3)}' for c in range(data.shape[1])]
        data = pd.DataFrame(data, columns=cols)
        data_.append(data)

    out_dir = op.join(cfg['save_dir'], 'preproc')
    if cfg['save_all']:
        for i, data in enumerate(data_):
            f_out = op.join(out_dir, cfg['f_base'] + f'_run-{i+1}_desc-preproc_conf.tsv')
            data.to_csv(f_out, sep='\t', index=False)

    # Concatenate DataFrames and save
    data = pd.concat(data_, axis=0, sort=True)
    f_out = op.join(out_dir, cfg['f_base'] + '_desc-preproc_conf.tsv')
    data.to_csv(f_out, sep='\t', index=False)

    ddict['preproc_conf'] = data
    return ddict


def preprocess_confs_noise_pool(ddict, cfg, logger):
    """ GLMdenoise style. """
    if np.unique(ddict['run_idx']).size < 2:
        raise ValueError("Cannot cross-validate with fewer than 2 runs")

    if cfg['hrf_model'] == 'kay':
        r2 = np.zeros((20, ddict['preproc_func'].shape[1]))
        to_iter = range(20)
    else:
        r2 = np.zeros((1, ddict['preproc_func'].shape[1]))
        to_iter = range(1)
    
    # Use linear regression with leaven-one-run-out CV
    model = LinearRegression(fit_intercept=False)
    cv = LeaveOneGroupOut()
    
    logger.info(f"Starting noise pool estimation using {len(to_iter)} HRF(s)")

    n_runs = np.unique(ddict['run_idx']).size
    for i in to_iter:
        Xs = []  # store runwise design matrix
        for run in range(n_runs):
            events = ddict['preproc_events'].query("run == (@run + 1)")
            tr = ddict['trs'][run]
            Y = ddict['preproc_func'][ddict['run_idx'] == run, :]
            ft = get_frame_times(tr, ddict, cfg, Y)
            X = create_design_matrix(tr, ft, events, hrf_model=cfg['hrf_model'], hrf_idx=i)
            X = X.iloc[:, :-1]  # remove intercept
            
            # Filter and make sure mean is 0
            X.loc[:, :] = hp_filter(X.to_numpy(), tr, ddict, cfg, standardize=False)
            X = X - X.mean(axis=0)
            Xs.append(X)

        # Concatenate design matrices
        X = pd.concat(Xs, axis=0).to_numpy()
        Y = ddict['preproc_func']  # already high-pass filtered

        # Cross-validation across runs + calculate noise pool
        r2[i, :] = cross_val_r2(model, X, Y, cv=cv, groups=ddict['run_idx'])
    
    # Get best score (across HRFs, if any) for each voxel
    r2_max = r2.max(axis=0)

    # Noise voxels are those with r2 < 0 (no need for additional signal-based)
    # threshold as in original GLMdenoise paper (because masked data)
    noisepool_idx = r2_max < 0
    logger.info(f"Noise pool has {noisepool_idx.sum()} voxels")

    decomp = PCA() if cfg['decomp'] == 'pca' else FastICA(max_iter=1000)
    cols = [f'decomp_{str(c+1).zfill(3)}' for c in range(cfg['n_comps'])] 
    
    # Do PCA per run
    data_ = []
    for i in range(n_runs):
        t_idx = ddict['run_idx'] == i
        Y = ddict['preproc_func'][t_idx, :]

        # Fit/transform + select only n_comps variables        
        data = decomp.fit_transform(Y[:, noisepool_idx])
        data = data[:, :cfg['n_comps']]

        # High-pass filter the confounds (again; not sure that's necessary)
        # and store
        data = hp_filter(data, ddict['trs'][i], ddict, cfg)
        data = pd.DataFrame(data, columns=cols)
        data_.append(data)

    # Save data to disk
    save_data(r2_max, cfg, ddict, par_dir='preproc', desc='max', dtype='r2', nii=True)
    save_data(r2_max, cfg, ddict, par_dir='preproc', desc='max', dtype='r2', nii=True)

    if r2.shape[0] > 1:  # also save R2 per HRF    
        save_data(r2, cfg, ddict, par_dir='preproc', run=None, desc='hrf', dtype='r2')
    
    # Save some more stuff
    if cfg['save_all']:
        for run, data in enumerate(data_):
            save_data(data, cfg, ddict, par_dir='preproc', run=run+1, desc='preproc',
                      dtype='conf', ext='tsv', skip_if_single_run=True)

    # Concatenate DataFrames and save
    data = pd.concat(data_, axis=0)
    save_data(data, cfg, ddict, par_dir='preproc', run=None, desc='preproc',
              dtype='conf', ext='tsv')

    ddict['preproc_conf'] = data

    return ddict


def preprocess_events(ddict, cfg, logger):
    """ Preprocesses event files. """
    
    data_ = []
    for i, event in enumerate(ddict['events']):
        first_match = [item if re.search(item[0], event, re.IGNORECASE) else 0 for item in cfg.get('skip_tr')][0]
        if first_match:
            skip_tr = first_match[1]
            match_data = [file if re.search(first_match[0], file, re.IGNORECASE) else 0 for file in ddict['funcs']][0]
            if 'fs' in cfg['space']:
                if cfg['iscift'] == 'y':
                    actual_tr = load_and_split_cifti(match_data, cfg['atlas_file'], cfg['left_id'],
                                                         cfg['right_id'], cfg['subc_id'])[1]
                else:
                    actual_tr = load_gifti(match_data, cfg)[1]
            else:
                actual_tr = nib.load(match_data).header['pixdim'][4]
            data = pd.read_csv(event, sep='\t')
            data['onset'] = data['onset'] - (actual_tr * skip_tr)
        else:
            data = pd.read_csv(event, sep='\t')
        if cfg['trial_filter'] is not None:
            data = data.copy().query(cfg['trial_filter'])

        # Check if necessary columns are there
        for col in ['onset', 'duration', 'trial_type']:  
            if col not in data.columns:
                raise ValueError(f"No column '{col}' in {event}!")

        # Check negative onsets
        neg_idx = data['onset'] < 0
        if neg_idx.sum() > 0:
            logger.warning(f"Removing {(neg_idx > 0).sum()} event(s) with a negative onset")
            data = data.loc[~neg_idx, :]
        
        # st = single trials
        if cfg['single_trial_id'] is not None:
            st_idx = data['trial_type'].str.contains(cfg['single_trial_id'])
            n_st = data.loc[st_idx, :].shape[0]
            
            # Setting a unique trial-type for single trials
            if cfg['signalproc_type'] == 'single-trial':
                data.loc[st_idx, 'trial_type'] = [f'{str(i).zfill(3)}_{s}' for i, s in enumerate(data.loc[st_idx, 'trial_type'])]
            else:
                # Not sure this should be the default, but set single trials to the
                # same condition if you want to do glmdenoise-style cross-validation stuff
                data.loc[st_idx, 'trial_type'] = 'stim'
            
            # Some bookkeeping (sorting and stuff)
            n_other = data.loc[~st_idx, 'trial_type'].unique().size
            sort_cols = ['onset', 'duration', 'trial_type', 'run']
            other_cols = [col for col in data.columns if col not in sort_cols]
            data = data.loc[:, sort_cols + other_cols]
            logger.info(f"Found {n_st} single trials and {n_other} other conditions for run {i+1}")
        else:
            conds = data['trial_type'].unique()
            to_print = sorted(conds) if len(conds) < 10 else sorted(conds)[:10]
            logger.info(f"Found {conds.size} conditions for run {i+1}: {to_print}")

        data['run'] = i+1
        first_cols = ['onset', 'duration', 'trial_type', 'run']
        data = data.loc[:, first_cols + [c for c in data.columns if c not in first_cols]]
        data_.append(data)

    # Save stuff
    if cfg['save_all']:
        for run, data in enumerate(data_):
            save_data(data, cfg, ddict, par_dir='preproc', run=run+1, desc='preproc',
                      dtype='events', ext='tsv', skip_if_single_run=True)

    # Adjust onsets for concatenated events file
    #for run in np.unique(ddict['run_idx']).astype(int):
    #    prev_run = ddict['run_idx'] < run
    #    data_[run].loc[:, 'onset'] = data_[run].loc[:, 'onset'] + prev_run.sum() * ddict['tr']

    # Concatenate events into one big DataFrame + save
    data = pd.concat(data_, axis=0)
    data.index = range(data.shape[0])
    if data.loc[:, ['onset', 'duration', 'trial_type']].isna().values.any() > 0:
        n_rows = data.shape[0]
        to_keep = data.loc[:, ['onset', 'duration', 'trial_type']].dropna(how='any', axis=0).index
        data = data.loc[to_keep, :]
        logger.warn(f"Removed {n_rows - data.shape[0]} rows containing NaNs")

    conditions = sorted(data['trial_type'].unique().tolist())
    for run in data['run'].unique():
        these_con = sorted(data.query("run == @run")['trial_type'].unique().tolist())
        if not these_con == conditions and cfg['noiseproc_type'] == 'between':
            logger.warn(
                f"Conditions are not the same across runs! "
                "This is a problem from GLMdenoise style analyses"
            )
            break

    save_data(data, cfg, ddict, par_dir='preproc', run=None, desc='preproc', dtype='events', ext='tsv')

    # Print some useful info about single-trials
    if cfg['single_trial_id'] is not None: 
        st_idx = data['trial_type'].str.contains(cfg['single_trial_id'])
        n_st = data.loc[st_idx, :].shape[0]
        other = data.loc[~st_idx, 'trial_type'].unique().tolist()
        n_other = len(other)
        logger.info(
            f"Found {data.shape[0]} events across {i+1} runs, of which "
            f"{n_st} single trials + {n_other} other conditions {other}"
        )
    ddict['preproc_events'] = data
    return ddict


def load_preproc_data(ddict, cfg):
    """ Loads preprocessed data. """
    in_dir = op.join(cfg['save_dir'], 'preproc')
    f_base = cfg['f_base'] + '_desc-preproc_'
    f_in = op.join(in_dir, f_base)

    # Load in mask
    if 'fs' in cfg['space']:
        ddict['mask'] = None 
    else:
        ddict['mask'] = nib.load(f_in + 'mask.nii.gz')

    # Load in data (bold data always as npy, because fast/efficient)
    ddict['preproc_func'] = np.load(f_in + 'bold.npy')
    
    # Load in TRs (inefficient; maybe save/load as yaml?)
    if 'fs' in cfg['space']:
        if cfg['iscift'] == 'y':
            ddict['trs'] = [load_and_split_cifti(f, cfg['atlas_file'], cfg['left_id'],
                                                 cfg['right_id'], cfg['subc_id'])[1] for f in ddict['funcs']]
        else:
            ddict['trs'] = [load_gifti(f, cfg)[1] for f in ddict['funcs']]  # quite inefficient
    else:
        ddict['trs'] = [nib.load(f).header['pixdim'][4] for f in ddict['funcs']]

    # Load conf
    ddict['preproc_conf'] = pd.read_csv(f_in + 'conf.tsv', sep='\t')

    # Load events (but only if we want to do signalproc)
    if not cfg['skip_signalproc']:
        ddict['preproc_events'] = pd.read_csv(f_in + 'events.tsv', sep='\t')
    else:
        ddict['preproc_events'] = None

    # Load run index (which time point belongs to which run?)
    ddict['run_idx'] = np.load(op.join(in_dir, f"task-{cfg['c_task']}_run_idx.npy"))
    
    return ddict