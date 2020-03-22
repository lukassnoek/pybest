import io
import os
import os.path as op
import logging
import numpy as np
import nibabel as nib
from tqdm import tqdm
from glob import glob


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-8s] [%(levelname)-7.7s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('pybest')


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    
    def flush(self):
        self.logger.log(self.level, self.buf)

tqdm_out = TqdmToLogger(logger, level=logging.INFO)


def check_parameters(space, tr):

    if 'fs' in space and tr is None:
        raise ValueError("TR (--tr) needs to be set when using surface data (--space fs*)!")


def set_defaults(bids_dir, out_dir, fprep_dir, ricor_dir, work_dir, subject, logger):

    if not op.isdir(bids_dir):
        raise ValueError(f"BIDS directory {bids_dir} does not exist!")

    logger.info(f"Working on BIDS directory {bids_dir}")

    if out_dir is None:  # Set default out_dir
        out_dir = op.join(bids_dir, 'derivatives', 'pybest')
        if not op.isdir(out_dir):
            os.makedirs(out_dir)

        logger.info(f"Setting output directory to {out_dir}")

    if fprep_dir is None:
        fprep_dir = op.join(bids_dir, 'derivatives', 'fmriprep')
        if not op.isdir(fprep_dir):
            raise ValueError(f"Fmriprep directory {fprep_dir} does not exist.")

        logger.info(f"Setting Fmriprep directory to {fprep_dir}")

    if ricor_dir is None:
        ricor_dir = op.join(bids_dir, 'derivatives', 'physiology')
        if not op.isdir(ricor_dir):
            ricor_dir = None
            logger.info("No RETROICOR directory, so assuming no physio data.")
    
    if ricor_dir is not None:
        logger.info(f"Setting RETROICOR directory to {ricor_dir}")

    if work_dir is None:
        work_dir = op.join(out_dir, 'work')
        if not op.isdir(work_dir):
            os.makedirs(work_dir)

        logger.info(f"Setting working directory to {work_dir}")

    if not subject:
        subject = None

    return bids_dir, out_dir, fprep_dir, ricor_dir, work_dir, subject


def find_exp_parameters(bids_dir, fprep_dir, subject, session, task):

    # Use all possible participants if not provided
    if subject is None:
        subject = [
            op.basename(s).split('-')[1] for s in
            sorted(glob(op.join(fprep_dir, 'sub-*')))
            if op.isdir(s)
        ]
        logger.info(f"Found {len(subject)} participant(s)")
    else:
        # Use a list by default
        subject = [subject]

    # Use all sessions if not provided
    if session is None:
        session = []
        for this_sub in subject:
            these_ses = [
                op.basename(s).split('-')[1] for s in
                sorted(glob(op.join(fprep_dir, f'sub-{this_sub}', 'ses-*')))
                if op.isdir(s)
            ]
            session.append(these_ses)
            logger.info(f"Found {len(these_ses)} session(s) for sub-{this_sub}")
    else:
        session = [session] * len(subject)

    # Use all tasks if no explicit task is provided
    if task is None:
        task = []
        for this_sub, these_ses in zip(subject, session):
            these_task = []
            for this_ses in these_ses:
                
                tmp = glob(op.join(
                    bids_dir,
                    f'sub-{this_sub}',
                    f'ses-{this_ses}',
                    'func',
                    f'*_events.tsv'
                ))

                these_ses_task = list(set(
                    [op.basename(f).split('task-')[1].split('_')[0] for f in tmp]
                ))
        
                these_task.append(these_ses_task)
                logger.info(f"Found {len(these_ses_task)} task(s) for sub-{this_sub} and ses-{this_ses}")

            task.append(these_task)
    else:
        task = [[task] * len(session)] * len(subject)

    return subject, session, task


def find_data(sub, ses, task, space,  hemi, bids_dir, fprep_dir, ricor_dir):

    # Set right "identifier" depending on fsaverage* or volumetric space
    space_idf = f'hemi-{hemi}.func.gii' if 'fs' in space else 'desc-preproc_bold.nii.gz'


    # Gather funcs, confs, tasks
    funcs = sorted(glob(op.join(
        fprep_dir, f'sub-{sub}', f'ses-{ses}', 'func', f'*task-{task}_*_space-{space}_{space_idf}'
    )))
    confs = sorted(glob(op.join(
        fprep_dir, f'sub-{sub}', f'ses-{ses}', 'func', f'*desc-confounds_regressors.tsv'
    )))
    events = sorted(glob(op.join(
        bids_dir, f'sub-{sub}', f'ses-{ses}', 'func', f'*task-{task}_*_events.tsv'
    )))
    if not all(len(funcs) == len(tmp) for tmp in [confs, events]):
        raise ValueError(
            f"Found unequal number of funcs ({len(funcs)}), confs ({len(confs)}), and events ({len(events)})."
        )
    logger.info(f"Found {len(funcs)} runs for task {task}")

    # Also find retroicor files
    if ricor_dir is not None:
        ricors = sorted(glob(op.join(
            ricor_dir, f'sub-{sub}', f'ses-{ses}', 'physio', f'*task-{task}_*_regressors.tsv'
        )))
        logger.info(f"Found {len(ricors)} RETROICOR files for task {task}")
    else:
        ricors = None

    if 'fs' not in space:
        fname = f'sub-{sub}_label-GM_probseg.nii.gz'
        gm_prob = op.join(fprep_dir, f'sub-{sub}', 'anat', fname)
    else:
        gm_prob = None

    return funcs, confs, events, ricors, gm_prob


def _load_gifti(f):
    """ Load gifti array. """
    f_gif = nib.load(f)
    return np.vstack([arr.data for arr in f_gif.darrays])
