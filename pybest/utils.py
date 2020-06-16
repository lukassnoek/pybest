import io
import os
import click
import os.path as op
import numpy as np
import nibabel as nib
from tqdm import tqdm
from glob import glob
from nilearn import plotting, signal
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.stats.first_level_model import run_glm
from sklearn.linear_model import LinearRegression


def check_parameters(cfg, logger):
    """ Checks parameter settings and raises errors in case of
    incompatible parameters. """
    
    if 'fs' in cfg['space'] and cfg['tr'] is None:
        raise ValueError("TR (--tr) needs to be set when using surface data (--space fs*)!")

    if cfg['single_trial_id'] is None:
        logger.warn("No single-trial-id found; skipping signalproc!")
        cfg['skip_signalproc'] = True

    if cfg['uncorrelation'] and cfg['single_trial_model'] == 'lss':
        raise ValueError("Cannot use uncorrelation in combination with LSS.")


def set_defaults(cfg, logger):
    """ Sets default inputs. """
    if not op.isdir(cfg['bids_dir']):
        raise ValueError(f"BIDS directory {cfg['bids_dir']} does not exist!")

    logger.info(f"Using BIDS directory {cfg['bids_dir']}")

    if cfg['out_dir'] is None:  # Set default out_dir
        cfg['out_dir'] = op.join(cfg['bids_dir'], 'derivatives', 'pybest')
        if not op.isdir(cfg['out_dir']):
            os.makedirs(cfg['out_dir'], exist_ok=True)

        logger.info(f"Setting output directory to {cfg['out_dir']}")

    if cfg['fprep_dir'] is None:
        cfg['fprep_dir'] = op.join(cfg['bids_dir'], 'derivatives', 'fmriprep')
        logger.info(f"Setting Fmriprep directory to {cfg['fprep_dir']}")

        if not op.isdir(cfg['fprep_dir']):
            raise ValueError(f"Fmriprep directory {cfg['fprep_dir']} does not exist.")

    if cfg['ricor_dir'] is None:
        cfg['ricor_dir'] = op.join(cfg['bids_dir'], 'derivatives', 'physiology')
        if not op.isdir(cfg['ricor_dir']):
            cfg['ricor_dir'] = None
            logger.info("No RETROICOR directory, so assuming no physio data.")
    else:
        logger.info(f"Setting RETROICOR directory to {cfg['ricor_dir']}")

    if cfg['session'] is None:
        logger.warning(f"No session identifier given; assuming a single session.")

    if cfg['gm_thresh'] == 0:
        cfg['gm_thresh'] = None

    if not cfg['subject']:
        cfg['subject'] = None

    return cfg


def find_exp_parameters(cfg, logger):
    """ Extracts experimental parameters. """
    # Use all possible participants if not provided
    if cfg['subject'] is None:
        cfg['subject'] = [
            op.basename(s).split('-')[1] for s in
            sorted(glob(op.join(cfg['fprep_dir'], 'sub-*')))
            if op.isdir(s)
        ]
        logger.info(f"Found {len(cfg['subject'])} participant(s) {cfg['subject']}")
    else:
        # Use a list by default
        cfg['subject'] = [cfg['subject']]

    # Use all sessions if not provided
    if cfg['session'] is None:
        cfg['session'] = []
        for this_sub in cfg['subject']:
            these_ses = [
                op.basename(s).split('-')[1] for s in
                sorted(glob(op.join(cfg['fprep_dir'], f'sub-{this_sub}', 'ses-*')))
                if op.isdir(s)
            ]
            logger.info(f"Found {len(these_ses)} session(s) for sub-{this_sub} {these_ses}")
            these_ses = [None] if not these_ses else these_ses
            cfg['session'].append(these_ses)
    else:
        cfg['session'] = [cfg['session']] * len(cfg['subject'])

    # Use all tasks if no explicit task is provided
    if cfg['task'] is None:
        cfg['task'] = []
        for this_sub, these_ses in zip(cfg['subject'], cfg['session']):
            these_task = []
            for this_ses in these_ses:
                if this_ses is None:  # only single session!
                    tmp = glob(op.join(
                        cfg['fprep_dir'],
                        f'sub-{this_sub}',
                        'func',
                        f"*space-{cfg['space']}*_desc-preproc_bold.nii.gz"
                    ))
                else:
                    tmp = glob(op.join(
                        cfg['fprep_dir'],
                        f'sub-{this_sub}',
                        f'ses-{this_ses}' ,
                        'func',
                        f"*space-{cfg['space']}*_desc-preproc_bold.nii.gz"
                    ))
                
                these_ses_task = list(set(
                    [op.basename(f).split('task-')[1].split('_')[0] for f in tmp]
                ))
        
                these_task.append(these_ses_task)
                
                to_add = "" if this_ses is None else f"and ses-{this_ses}" 
                msg = f"Found {len(these_ses_task)} task(s) for sub-{this_sub} {to_add} {these_ses_task}"

                logger.info(msg)

            cfg['task'].append(these_task)
    else:
        all_ses_tasks = []
        for this_sub, these_ses in zip(cfg['subject'], cfg['session']):
            these_task = []
            for this_ses in these_ses:
                if this_ses is None:
                    tmp = glob(op.join(
                        cfg['fprep_dir'],
                        f'sub-{this_sub}',
                        'func',
                        f"*task-{cfg['task']}*_space-{cfg['space']}*_desc-preproc_bold.nii.gz"
                    ))
                else:
                    tmp = glob(op.join(
                        cfg['fprep_dir'],
                        f'sub-{this_sub}',
                        f'ses-{this_ses}',
                        'func',
                        f"*task-{cfg['task']}*_space-{cfg['space']}*_desc-preproc_bold.nii.gz"
                    ))
                if tmp:
                    these_task.append([cfg['task']])
                else:
                    these_task.append([None])
            all_ses_tasks.append(these_task)
        
        cfg['task'] = all_ses_tasks

    return cfg


def find_data(cfg, logger):
    """ Finds all data for a given subject/session/task/space/hemi. """
    # Set right "identifier" depending on fsaverage* or volumetric space
    sub, ses, task, hemi, space = cfg['c_sub'], cfg['c_ses'], cfg['c_task'], cfg['hemi'], cfg['space']
    space_idf = f'hemi-{hemi}.func.gii' if 'fs' in space else 'desc-preproc_bold.nii.gz'

    # Gather funcs, confs, tasks
    fprep_dir = cfg['fprep_dir']
    if cfg['c_ses'] is None:
        ffunc_dir = op.join(fprep_dir, f'sub-{sub}', 'func')
    else:
        ffunc_dir = op.join(fprep_dir, f'sub-{sub}', f'ses-{ses}', 'func')

    funcs = sorted(glob(op.join(ffunc_dir, f'*task-{task}*_space-{space}_{space_idf}')))
    confs = sorted(glob(op.join(ffunc_dir, f'*task-{task}*_desc-confounds_regressors.tsv')))

    bids_dir = cfg['bids_dir']
    if cfg['c_ses'] is None:
        bfunc_dir = op.join(bids_dir, f'sub-{sub}', 'func')
    else:
        bfunc_dir = op.join(bids_dir, f'sub-{sub}', f'ses-{ses}', 'func')

    events = sorted(glob(op.join(bfunc_dir, f'*task-{task}*_events.tsv')))

    if len(events) == 0:
        logger.warning("Did not find event files! Going to assume there's no task involved.")
        events = None
        to_check = [confs]
    else:
        to_check = [confs, events]

    if not all(len(funcs) == len(tmp) for tmp in to_check):
        msg = f"Found unequal number of funcs ({len(funcs)}) and confs ({len(confs)})"
        if events is not None:
            msg += f" and events ({len(events)})"
        
        raise ValueError(msg)
    
    logger.info(f"Found {len(funcs)} runs for task {task}")

    # Also find retroicor files
    ricor_dir = cfg['ricor_dir']
    if ricor_dir is not None:
        ricors = sorted(glob(op.join(
            ricor_dir, f'sub-{sub}', f'ses-{ses}', 'physio', f'*task-{task}_*_regressors.tsv'
        )))
        logger.info(f"Found {len(ricors)} RETROICOR files for task {task}")
    else:
        ricors = None

    if 'fs' not in space and cfg['gm_thresh'] is not None:  # volumetric files
        space_idf = '' if space == 'T1w' else f'_space-{space}'
        fname = f'sub-{sub}{space_idf}_label-GM_probseg.nii.gz'
        gm_prob = op.join(fprep_dir, f'sub-{sub}', 'anat', fname)
    else:
        gm_prob = None

    ddict = dict(
        funcs=funcs, confs=confs, events=events,
        ricors=ricors, gm_prob=gm_prob
    )

    if cfg['tr'] is None:
        tr = np.round(nib.load(funcs[0]).header['pixdim'][4], 3)
        logger.warning(f"TR is not set; using TR from first func ({tr:.3f} sec.)")

    # Store TR in data dict (maybe should use cfg?)
    ddict['tr'] = tr

    return ddict


def _load_gifti(f):
    """ Load gifti array. """
    f_gif = nib.load(f)
    return np.vstack([arr.data for arr in f_gif.darrays])


def get_frame_times(ddict, cfg, Y):
    """ Computes frame times for a particular time series (and TR). """
    tr = ddict['tr']
    n_vol = Y.shape[0]
    st_ref = cfg['slice_time_ref']
    ft = np.linspace(st_ref * tr, n_vol * tr + st_ref * tr, n_vol, endpoint=False)
    return ft
  

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


@click.command()
@click.argument('file')
@click.option('--hemi', default='L', type=click.Choice(['L', 'R']), required=False)
@click.option('--space', default='fsaverage6', type=click.Choice(['fsaverage', 'fsaverage5', 'fsaverage6']), required=False)
@click.option('--fs-dir', default=None, required=False)
@click.option('--threshold', default=0., type=click.FLOAT, required=False)
def view_surf(file, hemi, space, fs_dir, threshold):
    """ Utility command to quickly view interactive surface in your browser. 
    
    file : str
        Path to numpy file with vertex data
    hemi : str
        Hemifield; either L or R
    space : str
        Space of vertices (fsaverage[,5,6])
    fs_dir : str
        Directory with space template (mutually exclusive with `space` param)
    threshold : float
        Minimum value to display
    """
    if fs_dir is not None:
        mesh = op.join(fs_dir, 'surf', f"{hemi.lower()}h.inflated")
        bg = op.join(fs_dir, 'surf', f"{hemi.lower()}h.sulc")
    else:
        hemi = 'left' if hemi == 'L' else 'right'
        fs = fetch_surf_fsaverage(mesh=space)
        mesh = fs[f"infl_{hemi}"]
        bg = fs[f"sulc_{hemi}"]
        
    dat = np.load(file)
    display = plotting.view_surf(
        surf_mesh=mesh,
        surf_map=dat,
        bg_map=bg,
        threshold=threshold
    )
    display.open_in_browser()


def get_run_data(ddict, run, func_type='preproc'):
    """ Get the data for a specific run. """
    t_idx = ddict['run_idx'] == run  # timepoint index
    func = ddict[f'{func_type}_func'][t_idx, :].copy()
    conf = ddict['preproc_conf'].copy().loc[t_idx, :].to_numpy()
    if ddict['preproc_events'] is not None:
        events = ddict['preproc_events'].copy().query("run == (@run + 1)")
    else:
        events = None
    # I think we need an explicit copy here (not sure)
    return func, conf, events


def yield_glm_results(vox_idx, Y, X, conf, run, ddict, cfg, noise_model='ols'):
    model = LinearRegression(fit_intercept=False)
    opt_n_comps = ddict['opt_noise_n_comps'][run, :]
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
        X_n = conf[:, :this_n_comps]
        this_X = X.copy()
        this_X.iloc[:, :] = this_X.to_numpy() - model.fit(X_n, this_X.to_numpy()).predict(X_n)
        
        # Set max to 1 (not sure whether I should actually do this)
        this_X.iloc[:, :-1] = this_X.iloc[:, :-1] / this_X.iloc[:, :-1].max(axis=0)
        
        # Refit model on all data this time and remove fitted values
        labels, results = run_glm(Y[:, this_vox_idx], this_X.to_numpy(), noise_model=noise_model)
        yield this_vox_idx, this_X, labels, results