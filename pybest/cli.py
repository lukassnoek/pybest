import click
import os.path as op
from glob import glob
from .util import logger, preprocess, Dataset
from .noise_model import optimize_noise_model
from .signal_model import optimize_signal_model


@click.command()
@click.argument('fprep_dir')
@click.argument('out_dir', default=None, required=False)
@click.option('--participant-label', default=None, required=False)
@click.option('--session', default=None, required=False)
@click.option('--task', default=None)
@click.option('--space', default='T1w')
def main(fprep_dir, out_dir, participant_label, session, task, space):
    """ Main API of pybest. """

    logger.info('Starting pybest!')

    if out_dir is None:  # Set default out_dir
        out_dir = op.join(op.dirname(fprep_dir), 'pybest')

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
        session = [session]

    # Use all tasks if not provided
    if task is None:
        for participant, this_ses in zip(participant_label, session):
            for ses in this_ses:
                tmp = glob(op.join(
                    fprep_dir,
                    f'sub-{participant}',
                    f'ses-{ses}',
                    'func',
                    f'*_events.tsv'
                ))

                task = list(set(
                    [op.basename(f).split('task-')[1].split('_')[0] for f in tmp]
                ))
                logger.info(f"Found {len(task)} tasks for sub-{participant} and ses-{ses}")

    # Find files, construct Dataset obj

    # Preprocess

    # Noise model

    # Signal model

if __name__ == '__main__':

    main()