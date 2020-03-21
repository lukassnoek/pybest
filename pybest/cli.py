import click
import os.path as op

@click.argument('fprep_dir')
@click.argument('out_dir', required=False)
@click.option('--participant-label')
@click.option('--task', default='*')
@click.option('--space', default='T1w')
def main(fprep_dir, out_dir, participant_label, task, space):
    """ Main API of pybest. """
    if out_dir is None:
        out_dir = op.join(op.dirname(fprep_dir), 'pybest')

    # Housekeeping

    pybest_main()

if __name__ == '__main__':

    main()