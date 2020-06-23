import os
import click
import os.path as op
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from nilearn import plotting, signal, masking
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.stats.first_level_model import run_glm
from sklearn.linear_model import LinearRegression
from nilearn.glm.first_level.experimental_paradigm import check_events
from nilearn.glm.first_level.design_matrix import make_first_level_design_matrix
from nilearn.glm.first_level.hemodynamic_models import _sample_condition, _resample_regressor
from nilearn.glm.first_level.design_matrix import _cosine_drift as dct_set

from .constants import HRFS_HR


def load_gifti(f, return_tr=True):
    """ Load gifti array. """
    f_gif = nib.load(f)
    data = np.vstack([arr.data for arr in f_gif.darrays])
    tr = float(f_gif.darrays[0].get_metadata()['TimeStep'])
    if return_tr:
        return data, tr
    else:
        return data


def save_data(data, cfg, ddict, par_dir, run, desc, dtype, ext=None, skip_if_single_run=False):
    """ Saves data as either numpy files (for fs* space data) or
    gzipped nifti (.nii.gz; for volumetric data).

    Parameters
    ----------
    data : np.ndarray
        Either a 1D (voxels,) or 2D (observations x voxels) array
    cfg : dict
        Config dictionary
    par_dir : str
        Name of parent directory ('preproc', 'denoising', 'best')
    run : int/None
        Run index (if None, assumed to be a single run)
    desc : str
        Description string (desc-{desc})
    dtype : str
        Type of data (_{dtype}.{npy,nii.gz})
    ext : str
        Extension (to determine how to save the data). If None, assuming
        fMRI data.
    """
    
    if data is None:
        return None

    if skip_if_single_run:
        if len(ddict['funcs']) == 1:
            return None

    save_dir = op.join(cfg['save_dir'], par_dir)
    if not op.isdir(save_dir):
        os.makedirs(save_dir)

    if run is None:
        f_out = op.join(save_dir, cfg['f_base'] + f'_desc-{desc}_{dtype}')
    else:
        f_out = op.join(save_dir, cfg['f_base'] + f'_run-{run}_desc-{desc}_{dtype}')

    if ext == 'tsv':
        data.to_csv(f_out + '.tsv', sep='\t', index=False)
        return None

    if 'fs' in cfg['space']:  # surface
        np.save(f_out + '.npy', data)
    else:  # volume
        if isinstance(data, nib.Nifti1Image):
            data.to_filename(f_out + '.nii.gz')
        else:
            masking.unmask(data, ddict['mask']).to_filename(f_out + '.nii.gz')


def hp_filter(data, tr, ddict, cfg, standardize=True):
    """ High-pass filter (DCT or Savitsky-Golay). """

    n_vol = data.shape[0]
    st_ref = cfg['slice_time_ref']
    ft = np.linspace(st_ref * tr, n_vol * (tr + st_ref), n_vol, endpoint=False)

    # Create high-pass filter and clean
    if cfg['high_pass_type'] == 'dct':
        hp_set = dct_set(cfg['high_pass'], ft)
        data = signal.clean(data, detrend=False, standardize=standardize, confounds=hp_set)
    else:  # savgol, hardcode polyorder
        window = int(np.round((1 / cfg['high_pass']) / tr))
        data -= savgol_filter(data, window_length=window, polyorder=2, axis=0)
        if standardize:
            data = signal.clean(data, detrend=False, standardize=standardize)

    return data


def get_frame_times(tr, ddict, cfg, Y):
    """ Computes frame times for a particular time series (and TR). """
    n_vol = Y.shape[0]
    st_ref = cfg['slice_time_ref']
    frame_times = np.linspace(
        st_ref * tr,
        n_vol * tr + st_ref * tr,
        n_vol,
        endpoint=False
    )
    return frame_times


def get_param_from_glm(name, labels, results, dm, time_series=False, predictors=False):
    """ Get parameters from a fitted nistats GLM. """
    if predictors and time_series:
        raise ValueError("Cannot get predictors and time series.")

    if time_series:
        data = np.zeros((dm.shape[0], labels.size))
    elif predictors:
        data = np.zeros((dm.shape[1], labels.size))
    else:
        data = np.zeros_like(labels)

    for lab in np.unique(labels):
        data[..., labels == lab] = getattr(results[lab], name)

    return data


def create_design_matrix(tr, frame_times, events, hrf_model='kay', hrf_idx=None):
    """ Creates a design matrix based on a HRF from Kendrick Kay's set
    or a default one from Nistats. """
    
    # This is to keep oversampling consistent across hrf_models
    hrf_oversampling = 10
    design_oversampling = tr / (0.1 / hrf_oversampling)

    if hrf_model != 'kay':  # just use Nilearn
        return make_first_level_design_matrix(
            frame_times, events, drift_model=None, min_onset=0,
            oversampling=design_oversampling, hrf_model=hrf_model
        )

    if hrf_model == 'kay':
        
        if hrf_idx is None:  # 20 different DMs (based on different HRFs)
            to_iter = range(HRFS_HR.shape[1])
        else:
            to_iter = [hrf_idx]

        # dms will store all design matrices
        dms = []
        for hrf_idx in to_iter:
            hrf = HRFS_HR[:, hrf_idx]
            
            # To match the design oversampling, do it relative to tr
            trial_type, onset, duration, modulation = check_events(events)

            # Pre-allocate design matrix; note: columns are alphabetically sorted
            X = np.zeros((frame_times.size, np.unique(trial_type).size))
            uniq_trial_types = np.unique(trial_type)

            # Create separate regressor for each unique trial type
            for i, condition in enumerate(uniq_trial_types):
                condition_mask = (trial_type == condition)
                exp_condition = (
                    onset[condition_mask],
                    duration[condition_mask],
                    modulation[condition_mask]
                )
                # Create high resolution regressor/frame times
                hr_regressor, hr_frame_times = _sample_condition(
                    exp_condition, frame_times, design_oversampling, 0
                )
                
                # Convolve with HRF and downsample
                conv_reg = np.convolve(hr_regressor, hrf)[:hr_regressor.size]
                f = interp1d(hr_frame_times, conv_reg)
                X[:, i] = f(frame_times).T
            
            X /= X.max(axis=0)  # rescale to max = 1
            dm = pd.DataFrame(X, columns=uniq_trial_types, index=frame_times)
            dm['constant'] = 1  # and intercept/constant
            dms.append(dm)

        if len(dms) == 1:
            # Just return single design matrix
            dms = dms[0]

        return dms 


def get_run_data(ddict, run, func_type='preproc'):
    """ Get the data for a specific run. """
    
    t_idx = ddict['run_idx'] == run  # timepoint index
    func = ddict[f'{func_type}_func'][t_idx, :].copy()
    conf = ddict['preproc_conf'].copy().loc[t_idx, :].to_numpy()
    
    if ddict['preproc_events'] is not None:
        events = ddict['preproc_events'].copy().query("run == (@run + 1)")
    else:
        events = None
    
    return func, conf, events


def yield_glm_results(vox_idx, Y, X, conf, run, ddict, cfg):

    tr = ddict['trs'][run]
    opt_n_comps = ddict['opt_n_comps'][run, :].astype(int)
    nm = cfg['single_trial_noise_model']
    for this_n_comps in np.unique(opt_n_comps):  # loop across unique n comps
        # If n_comps is 0, then R2 was negative and we
        # don't want to denoise, so continue
        if this_n_comps == 0:
            continue

        # Find voxels that correspond to this_n_comps and intersect
        # with given voxel index
        this_vox_idx = opt_n_comps == this_n_comps
        this_vox_idx = np.logical_and(vox_idx, this_vox_idx)

        # Get confound matrix (X_n) and remove from design (this_X)
        C = conf[:, :this_n_comps]
        this_X = X.copy()
        if 'constant' in this_X.columns:
            X = X.drop('constant', axis=1)
        
        this_X.loc[:, :], Y = custom_clean(this_X, Y, C, tr, ddict, cfg, standardize=True)

        # Set max to 1 (not sure whether I should actually do this)
        this_X.iloc[:, :-1] = this_X.iloc[:, :-1] / \
            this_X.iloc[:, :-1].max(axis=0)

        
        # Refit model on all data this time and remove fitted values
        labels, results = run_glm(Y[:, this_vox_idx], this_X.to_numpy(), noise_model=nm)
        yield this_vox_idx, this_X, labels, results


def custom_clean(X, Y, C, tr, ddict, cfg, high_pass=True, clean_Y=True, standardize=True):
    """ High-passes (optional) and removes confounds (C) from both
    design matrix (X) and data (Y).

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe with design (timepoints x conditions)
    Y : np.ndarray
        2D numpy array (timepoints x voxels)
    C : pd.DataFrame
        Dataframe with confounds (timepoints x confound variables)
    high_pass : bool
        Whether to high-pass the data or not
    """

    if 'constant' in X.columns:
        X = X.drop('constant', axis=1)

    if high_pass:
        # Note to self: Y and C are already high-pass filtered
        X.loc[:, :] = hp_filter(X.to_numpy(), tr, ddict, cfg, standardize=False)
    
    X.loc[:, :] = signal.clean(X.to_numpy(), detrend=False, standardize=False)
    X = X - X.mean(axis=0)
    if clean_Y:
        Y = signal.clean(Y, detrend=False, confounds=C, standardize=standardize)

    return X, Y


@click.command()
@click.argument('file')
@click.option('--hemi', default=None, type=click.Choice(['L', 'R']), required=False)
@click.option('--space', default=None, type=click.Choice(['fsaverage', 'fsaverage5', 'fsaverage6']), required=False)
@click.option('--fs-dir', default=None, required=False)
@click.option('--threshold', default=0., type=click.FLOAT, required=False)
@click.option('--idx', default=None, type=click.INT, required=False)
def view_surf(file, hemi, space, fs_dir, threshold, idx):
    """ Utility command to quickly view interactive surface in your browser.

    file : str
        Path to numpy file (.npy) with vertex data
    hemi : str
        Hemifield; either L or R
    space : str
        Space of vertices (fsaverage[,5,6])
    fs_dir : str
        Directory with space template (mutually exclusive with `space` param)
    threshold : float
        Minimum value to display
    idx : int
        If data has multiple timepoints/observations, visualize the one corresponding to
        this index
    """

    if hemi is None:
        if 'hemi-' in file:
            hemi = file.split('hemi-')[1][0]
        else:
            raise ValueError(
                "Could not determine hemisphere from filename; "
                "set it explicitly using --hemi {L,R}"
            )

    if space is None and fs_dir is None:
        # Try to determine space from filename
        if 'space-' in file:
            space = file.split('space-')[1].split('_')[0]
        else:
            raise ValueError(
                "Could not determine space from filename; "
                "set it explicitly using --space or --fs-dir"
            )
    if fs_dir is not None:  # use data from specified Freesurfer dir
        mesh = op.join(fs_dir, 'surf', f"{hemi.lower()}h.inflated")
        bg = op.join(fs_dir, 'surf', f"{hemi.lower()}h.sulc")
    else:  # fetch template using Nilearn
        hemi = 'left' if hemi == 'L' else 'right'
        fs = fetch_surf_fsaverage(mesh=space)
        mesh = fs[f"infl_{hemi}"]
        bg = fs[f"sulc_{hemi}"]

    dat = np.load(file)
    if idx is not None:  # select volume
        dat = dat[idx, :]
    
    if dat.ndim > 1:
        raise ValueError("Data is 2D! Set --idx explicitly")

    display = plotting.view_surf(
        surf_mesh=mesh,
        surf_map=dat,
        bg_map=bg,
        threshold=threshold
    )
    display.open_in_browser()
