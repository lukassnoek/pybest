import os
import click
import os.path as op
import numpy as np
import nibabel as nib
from glob import glob
from .utils import logger, check_parameters, set_defaults
from .utils import find_exp_parameters, find_data
from .preproc import preprocess_funcs, preprocess_confs, preprocess_events
from .preproc import load_preproc_data
from .noise_model import run_noise_processing, load_denoised_data

@click.command()
@click.argument('bids_dir')
@click.argument('out_dir', default=None, required=False)
@click.argument('fprep_dir', default=None, required=False)
@click.argument('ricor_dir', default=None, required=False)
@click.argument('subject', nargs=-1, required=False)
@click.option('--work-dir', default=None, required=False)
@click.option('--start-from', type=click.Choice(['preproc', 'noiseproc', 'signalproc']), default='preproc', required=False)
@click.option('--session', default=None, required=False)
@click.option('--task', default=None)
@click.option('--space', default='T1w', show_default=True)
@click.option('--gm-thresh', default=0.9, show_default=True)  # maybe use a "mask" option
@click.option('--high-pass-type', type=click.Choice(['dct', 'savgol']), default='dct', show_default=True)
@click.option('--high-pass', default=0.01, show_default=True)
@click.option('--hemi', type=click.Choice(['L', 'R']), default='L', show_default=True)
@click.option('--tr', default=None, type=click.FLOAT, show_default=True)
@click.option('--decomp', default='pca', type=click.Choice(['pca', 'ica']), show_default=True)
@click.option('--ncomps', default=100, type=click.INT, show_default=True)
@click.option('--cv-repeats', default=2, type=click.INT, show_default=True)
@click.option('--cv-splits', default=5, type=click.INT, show_default=True)
@click.option('--nthreads', default=1, show_default=True)
def main(bids_dir, out_dir, fprep_dir, ricor_dir, subject, work_dir, start_from, session, task, space,
         gm_thresh, high_pass_type, high_pass, hemi, tr, decomp, ncomps, cv_repeats, cv_splits, nthreads):
    """ Main API of pybest. """

    ##### set + check parameters #####
    cfg = locals()
    cfg = set_defaults(cfg, logger)
    check_parameters(cfg, logger)

    ##### find data #####
    cfg = find_exp_parameters(cfg)

    ##### <start processing loop> #####
    for i, sub in enumerate(cfg['subject']):
        for ii, ses in enumerate(cfg['session'][i]):
            for task in cfg['task'][i][ii]:
                logger.info(f"Starting process for sub-{sub}, ses-{ses}, task-{task}")
                cfg['f_base'] = f"sub-{sub}_ses-{ses}_task-{task}"
                cfg['out_dir'] = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}')
                if not op.isdir(cfg['out_dir']):
                    os.makedirs(cfg['out_dir'])

                for key, value in [('sub', sub), ('ses', ses), ('task', task)]:
                    cfg[key] = value


                # ddict = data dictionary
                ddict = find_data(cfg, logger)
                
                if tr is None:
                    tr = np.round(nib.load(ddict['funcs'][0]).header['pixdim'][4], 3)
                    logger.info(f"TR is not set; extracted TR from first func is {tr:.3f}")

                # Store TR in data dict
                ddict['tr'] = tr

                if start_from == 'preproc':
                    ddict = preprocess_funcs(ddict, cfg, logger)
                    ddict = preprocess_confs(ddict, cfg, logger)                    
                    ddict = preprocess_events(ddict, cfg, logger)
                    
                # If we did preprocessing already ...
                if start_from == 'noiseproc':
                    ddict = load_preproc_data(ddict, cfg)
                
                # ... and didn't do noiseprocessing yet ...
                if not start_from == 'signalproc':
                    ddict = run_noise_processing(ddict, cfg, logger)
                else:
                    # If we did, load the denoised data
                    logger.info("Loading denoised data")
                    ddict = load_denoised_data(ddict, cfg)

                # DANIEL, YOU START HERE. YOU MAY ASSUME YOU HAVE THE FOLLOWING VARIABLES:
                # - func_data: 2D array (time x voxels)
                # - event_data: dataframe (time x pca-decomposed confounds)
                # - run_idx: 1D array (time)
                # - bookkeeping stuff, e.g., sub (01), ses (1), task (face), work_dir, etc


if __name__ == '__main__':
    main()