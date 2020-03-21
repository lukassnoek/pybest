import os
import click
import os.path as op
from glob import glob
from .util import logger, preprocess, Dataset
from .noise_model import optimize_noise_model
from .signal_model import optimize_signal_model


@click.command()
@click.argument('bids_dir')
@click.argument('out_dir', default=None, required=False)
@click.argument('fprep_dir', default=None, required=False)
@click.option('--participant-label', default=None, required=False)
@click.option('--session', default=None, required=False)
@click.option('--task', default=None)
@click.option('--space', default='T1w')
@click.option('--hemi', default='L')
def main(bids_dir, out_dir, fprep_dir, participant_label, session, task, space, hemi):
    """ Main API of pybest. """

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

    # Use all tasks if not provided
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

    space_idf = f'hemi-{hemi}.func.gii' if 'fs' in space else 'desc-preproc_bold.nii.gz'

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
                logger.info(f"Found {len(funcs)} runs of task {task}")


if __name__ == '__main__':

    main()