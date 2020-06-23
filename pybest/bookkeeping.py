import os
import os.path as op
from glob import glob


def check_parameters(cfg, logger):
    """ Checks parameter settings and raises errors in case of
    incompatible parameters. """

    if cfg['single_trial_id'] is None:
        logger.warn("No single-trial-id found! Treating trials as conditions")

    if cfg['uncorrelation'] and cfg['single_trial_model'] == 'lss':
        raise ValueError("Cannot use uncorrelation in combination with LSS.")

    if cfg['noise_source'] == 'noisepool' and cfg['skip_signalproc']:
        raise ValueError("Cannot skip signal proc when using noise-source == noisepool")

    if cfg['signalproc_type'] == 'glmdenoise' and not cfg['ignore_sessions']:
        logger.warn(
            f"It's recommended to pool data across sessions with GLMdenoise"
            "-style denoising in order to use all trials available! "
            "To do so, use --ingore-sessions"
        )


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
            raise ValueError(
                f"Fmriprep directory {cfg['fprep_dir']} does not exist.")

    if cfg['ricor_dir'] is None:
        cfg['ricor_dir'] = op.join(
            cfg['bids_dir'], 'derivatives', 'physiology')
        if not op.isdir(cfg['ricor_dir']):
            cfg['ricor_dir'] = None
            logger.info("No RETROICOR directory, so assuming no physio data.")
    else:
        logger.info(f"Setting RETROICOR directory to {cfg['ricor_dir']}")

    if cfg['session'] is None:
        logger.warning(
            f"No session identifier given; assuming a single session.")

    if cfg['gm_thresh'] == 0:
        cfg['gm_thresh'] = None

    if not cfg['subject']:
        cfg['subject'] = None

    return cfg


def find_exp_parameters(cfg, logger):
    """ Extracts experimental parameters. """

    hemi, space = cfg['hemi'], cfg['space']
    space_idf = f'hemi-{hemi}.func.gii' if 'fs' in space else 'desc-preproc_bold.nii.gz'

    # Use all possible participants if not provided
    if cfg['subject'] is None:
        cfg['subject'] = [
            op.basename(s).split('-')[1] for s in
            sorted(glob(op.join(cfg['fprep_dir'], 'sub-*')))
            if op.isdir(s)
        ]
        logger.info(
            f"Found {len(cfg['subject'])} participant(s) {cfg['subject']}")
    else:
        # Use a list by default
        cfg['subject'] = [cfg['subject']]

    # Use all sessions if not provided
    if cfg['session'] is None:
        cfg['session'] = []
        for this_sub in cfg['subject']:
            these_ses = [
                op.basename(s).split('-')[1] for s in
                sorted(
                    glob(op.join(cfg['fprep_dir'], f'sub-{this_sub}', 'ses-*')))
                if op.isdir(s)
            ]
            logger.info(
                f"Found {len(these_ses)} session(s) for sub-{this_sub} {these_ses}")
            these_ses = [None] if not these_ses else these_ses
            cfg['session'].append(these_ses)
    else:
        cfg['session'] = [[cfg['session']]] * len(cfg['subject'])
    
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
                        f"*space-{cfg['space']}*_{space_idf}"
                    ))
                else:
                    tmp = glob(op.join(
                        cfg['fprep_dir'],
                        f'sub-{this_sub}',
                        f'ses-{this_ses}',
                        'func',
                        f"*space-{cfg['space']}*_{space_idf}"
                    ))

                these_ses_task = list(set(
                    [op.basename(f).split('task-')[1].split('_')[0]
                     for f in tmp]
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
                        f"*task-{cfg['task']}*_space-{cfg['space']}*_{space_idf}"
                    ))
                else:
                    tmp = glob(op.join(
                        cfg['fprep_dir'],
                        f'sub-{this_sub}',
                        f'ses-{this_ses}',
                        'func',
                        f"*task-{cfg['task']}*_space-{cfg['space']}*_{space_idf}"
                    ))
                if tmp:
                    these_task.append([cfg['task']])
                else:
                    these_task.append([None])
            all_ses_tasks.append(these_task)
        
        cfg['task'] = all_ses_tasks

    # If --ignore-sessions, then "pool" all runs in a single session
    # (Maybe this should be default, anyway.)
    if cfg['ignore_sessions']:
        cfg['session'] = [[None] for _ in range(len(cfg['subject']))]
        cfg['task'] = [[[t for t in task[0]]] for task in cfg['task']]

    return cfg


def find_data(cfg, logger):
    """ Finds all data for a given subject/session/task/space/hemi. """
    
    # Set right "identifier" depending on fsaverage* or volumetric space
    sub, ses, task, hemi, space = cfg['c_sub'], cfg['c_ses'], cfg['c_task'], cfg['hemi'], cfg['space']
    if cfg['ignore_sessions']:
        ses = '*'  # wilcard for globbing across sessions

    # idf = identifier for files
    idf = f'hemi-{hemi}.func.gii' if 'fs' in space else 'desc-preproc_bold.nii.gz'

    # Gather funcs, confs, tasks
    fprep_dir = cfg['fprep_dir']
    if ses is None:
        ffunc_dir = op.join(fprep_dir, f'sub-{sub}', 'func')
    else:
        ffunc_dir = op.join(fprep_dir, f'sub-{sub}', f'ses-{ses}', 'func')

    funcs = sorted(glob(op.join(ffunc_dir, f'*task-{task}*_space-{space}_{idf}')))
    confs = sorted(glob(op.join(ffunc_dir, f'*task-{task}*_desc-confounds_regressors.tsv')))

    # Find event files, which should be in the BIDS dir
    bids_dir = cfg['bids_dir']
    if ses is None:
        bfunc_dir = op.join(bids_dir, f'sub-{sub}', 'func')
    else:
        bfunc_dir = op.join(bids_dir, f'sub-{sub}', f'ses-{ses}', 'func')

    events = sorted(glob(op.join(bfunc_dir, f'*task-{task}*_events.tsv')))

    if len(events) == 0:
        logger.warning(
            "Did not find event files! Going to assume there's no task involved."
        )
        events = None
        to_check = [confs]
    else:
        to_check = [confs, events]

    # Number of funcs and confs (and possibly events) should be the same
    if not all(len(funcs) == len(tmp) for tmp in to_check):
        msg = f"Found unequal number of funcs ({len(funcs)}) and confs ({len(confs)})"
        if events is not None:
            msg += f" and events ({len(events)})"

        raise ValueError(msg)

    logger.info(f"Found {len(funcs)} complete runs for task {task}")

    # Also find retroicor files
    ricor_dir = cfg['ricor_dir']
    if ricor_dir is not None:
        ricors = sorted(glob(op.join(
            ricor_dir, f'sub-{sub}', f'ses-{ses}', 'physio', f'*task-{task}*regressors.tsv'
        )))
        logger.info(f"Found {len(ricors)} RETROICOR files for task {task}")
    else:
        ricors = None

    # Find gray-matter probability file
    if 'fs' not in space and cfg['gm_thresh'] is not None:  # volumetric files
        space_idf = '' if space == 'T1w' else f'_space-{space}'
        fname = f'sub-{sub}{space_idf}_label-GM_probseg.nii.gz'
        gm_prob = op.join(fprep_dir, f'sub-{sub}', 'anat', fname)
    else:  # either surface data or using func brain mask
        gm_prob = None

    # ddict: data dictionary
    ddict = dict(
        funcs=funcs,
        confs=confs,
        events=events,
        ricors=ricors,
        gm_prob=gm_prob
    )

    return ddict
