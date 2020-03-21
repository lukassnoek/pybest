import os
import click
import os.path as op
from glob import glob
from .utils import logger
from .preproc import preprocess
from .noise_model import optimize_noise_model
from .signal_model import optimize_signal_model


@click.command()
@click.argument('bids_dir')
@click.argument('out_dir', default=None, required=False)
@click.argument('fprep_dir', default=None, required=False)
@click.argument('ricor_dir', default=None, required=False)
@click.argument('participant-label', nargs=-1, required=False)
@click.option('--session', default=None, required=False)
@click.option('--task', default=None)
@click.option('--space', default='T1w', show_default=True)
@click.option('--high-pass', default=0.1, show_default=True)
@click.option('--hemi', type=click.Choice(['L', 'R']), default='L', show_default=True)
@click.option('--tr', default=0.7, show_default=True)
@click.option('--nthreads', default=1, show_default=True)
def main(bids_dir, out_dir, fprep_dir, ricor_dir, participant_label, session, task,
         space, high_pass, hemi, tr, nthreads):
    """ Main API of pybest. """

    ##### <set defaults> #####
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

    if not participant_label:
        participant_label = None
    
    ##### </set defaults> #####

    ##### <gather data> #####
    # Very ugly code, but necessary 

    # Use all possible participants if not provided
    if participant_label is None:
        participant_label = [
            op.basename(s).split('-')[1] for s in
            sorted(glob(op.join(fprep_dir, 'sub-*')))
            if op.isdir(s)
        ]
        logger.info(f"Found {len(participant_label)} participant(s)")
    else:
        # Use a list by default
        participant_label = [participant_label]

    # Use all sessions if not provided
    if session is None:
        session = []
        for participant in participant_label:
            this_sess = [
                op.basename(s).split('-')[1] for s in
                sorted(glob(op.join(fprep_dir, f'sub-{participant}', 'ses-*')))
                if op.isdir(s)
            ]
            session.append(this_sess)
            logger.info(f"Found {len(this_sess)} session(s) for sub-{participant}")
    else:
        session = [session] * len(participant_label)

    # Use all tasks if no explicit task is provided
    if task is None:
        task = []
        for participant, this_ses in zip(participant_label, session):
            sub_tasks = []
            for ses in this_ses:
                
                tmp = glob(op.join(
                    bids_dir,
                    f'sub-{participant}',
                    f'ses-{ses}',
                    'func',
                    f'*_events.tsv'
                ))

                these_task = list(set(
                    [op.basename(f).split('task-')[1].split('_')[0] for f in tmp]
                ))
        
                sub_tasks.append(these_task)
                logger.info(f"Found {len(these_task)} task(s) for sub-{participant} and ses-{ses}")

            task.append(sub_tasks)
    else:
        task = [[task] * len(session)] * len(participant_label)

    ##### </gather data> #####

    # Set right "identifier" depending on fsaverage* or volumetric space
    space_idf = f'hemi-{hemi}.func.gii' if 'fs' in space else 'desc-preproc_bold.nii.gz'

    ##### <start processing loop> #####
    for i, participant in enumerate(participant_label):
        for ii, ses in enumerate(session[i]):
            for iii, task in enumerate(task[i][ii]):
                logger.info(f"Starting process for sub-{participant}, ses-{ses}, task-{task}")

                # Gather funcs, confs, tasks
                funcs = sorted(glob(op.join(
                    fprep_dir, f'sub-{participant}', f'ses-{ses}', 'func', f'*task-{task}_*_space-{space}_{space_idf}'
                )))
                confs = sorted(glob(op.join(
                    fprep_dir, f'sub-{participant}', f'ses-{ses}', 'func', f'*desc-confounds_regressors.tsv'
                )))
                events = sorted(glob(op.join(
                    bids_dir, f'sub-{participant}', f'ses-{ses}', 'func', f'*task-{task}_*_events.tsv'
                )))
                if not all(len(funcs) == len(tmp) for tmp in [confs, events]):
                    raise ValueError(
                        f"Found unequal number of funcs ({len(funcs)}), confs ({len(confs)}), and events ({len(events)})."
                    )
                logger.info(f"Found {len(funcs)} runs for task {task}")

                # Also find retroicor files
                if ricor_dir is not None:
                    ricors = sorted(glob(op.join(
                        ricor_dir, f'sub-{participant}', f'ses-{ses}', 'physio', f'*task-{task}_*_regressors.tsv'
                    )))
                    logger.info(f"Found {len(ricors)} RETROICOR files for task {task}")

                if 'fs' not in space:
                    fname = f'sub-{participant}_label-GM_probseg.nii.gz'
                    gm_prob = op.join(fprep_dir, f'sub-{participant}', 'anat', fname)
                else:
                    gm_prob = None

                ##### RUN PREPROCESSING #####
                data, run_idx = preprocess(funcs, mask=gm_prob, space=space, tr=tr, logger=logger)

    ##### </end processing loop> #####

if __name__ == '__main__':

    main()