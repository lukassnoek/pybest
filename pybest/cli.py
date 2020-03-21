import click
import os.path as op

@click.argument('fprep_dir')
@click.argument('out_dir')
@click.option('--participant-label')
def main(fprep_dir, out_dir, participant_label):
    """ Main API of pybest. """
    if out_dir is None:
        out_dir = op.join(op.dirname(fprep_dir), 'pybest')
ß
    pass  # run main function

if __name__ == '__main__':

    main()