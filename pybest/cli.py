import os
import click
import os.path as op
import numpy as np
import nibabel as nib
from glob import glob
from .utils import logger, check_parameters, set_defaults
from .utils import find_exp_parameters, find_data
from .preproc import preprocess_funcs, preprocess_confs, preprocess_events
from .preproc import load_preproc_data, save_preproc_data
from .noise_model import run_noise_processing, save_denoised_data, load_denoised_data

@click.command()
@click.argument('bids_dir')
@click.argument('out_dir', default=None, required=False)
@click.argument('fprep_dir', default=None, required=False)
@click.argument('ricor_dir', default=None, required=False)
@click.argument('subject', nargs=-1, required=False)
@click.option('--work-dir', default=None, required=False)
@click.option('--start-from', type=click.Choice(['preproc', 'noiseproc', 'signalproc']), default='preproc', required=False)
@click.option('--denoising-strategy', type=click.Choice(['dummy']), default='dummy', required=False)
@click.option('--session', default=None, required=False)
@click.option('--task', default=None)
@click.option('--space', default='T1w', show_default=True)
@click.option('--gm-thresh', default=0.9, show_default=True)
@click.option('--high-pass-type', type=click.Choice(['dct', 'savgol']), default='dct', show_default=True)
@click.option('--high-pass', default=0.1, show_default=True)
@click.option('--savgol-order', default=4, show_default=True)
@click.option('--hemi', type=click.Choice(['L', 'R']), default='L', show_default=True)
@click.option('--tr', default=None, type=click.FLOAT, show_default=True)
@click.option('--nthreads', default=1, show_default=True)
def main(bids_dir, out_dir, fprep_dir, ricor_dir, subject, work_dir, start_from, denoising_strategy,
         session, task, space, gm_thresh, high_pass_type, high_pass, savgol_order, hemi, tr, nthreads):
    """ Main API of pybest. """

    ##### set defaults #####
    bids_dir, out_dir, fprep_dir, ricor_dir, work_dir, subject = set_defaults(
        bids_dir, out_dir, fprep_dir, ricor_dir, work_dir, subject, logger
    )

    check_parameters(space, tr)

    ##### find data #####
    subject, session, task = find_exp_parameters(bids_dir, fprep_dir, subject, session, task)

    ##### <start processing loop> #####
    for i, sub in enumerate(subject):
        for ii, ses in enumerate(session[i]):
            for iii, task in enumerate(task[i][ii]):
                logger.info(f"Starting process for sub-{sub}, ses-{ses}, task-{task}")

                funcs, confs, events, ricors, gm_prob = find_data(
                    sub, ses, task, space,  hemi, bids_dir, fprep_dir, ricor_dir
                )

                if tr is None:
                    tr = np.round(nib.load(funcs[0]).header['pixdim'][4], 3)
                    logger.info(f"TR is not set; extracted TR from first func is {tr}")

                if start_from == 'preproc':
                    func_data, run_idx, mask = preprocess_funcs(
                        funcs,
                        gm_prob,
                        space,
                        high_pass_type,
                        high_pass,
                        savgol_order,
                        gm_thresh,
                        tr,
                        logger
                    )
                    
                    conf_data = preprocess_confs(
                        confs,
                        ricors,
                        high_pass_type,
                        high_pass,
                        savgol_order,
                        tr,
                        logger
                    )
                    
                    event_data = preprocess_events(events, logger)
                    
                    logger.info("Saving preprocessed data")
                    save_preproc_data(
                        sub, ses, task, func_data, conf_data,
                        event_data, mask, run_idx, work_dir
                    )
                
                # If we did preprocessing already ...
                if start_from == 'noiseproc':
                    func_data, conf_data, event_data, mask, run_idx = load_preproc_data(
                        sub, ses, task, work_dir
                    )
                
                # ... and didn't do noiseprocessing yet ...
                if not start_from == 'signalproc':
                    func_data = run_noise_processing(
                        func_data,
                        conf_data,
                        run_idx,
                        denoising_strategy,
                        logger
                    )
                    logger.info("Saving denoised data")
                    save_denoised_data(sub, ses, task, func_data, mask, work_dir)
                else:
                    # If we did, load the denoised data
                    logger.info("Loading denoised data")
                    func_data, event_data, mask, run_idx = load_denoised_data(
                        sub, ses, task, work_dir
                    )

                # DANIEL, YOU START HERE. YOU MAY ASSUME YOU HAVE THE FOLLOWING VARIABLES:
                # - func_data: 2D array (time x voxels)
                # - event_data: dataframe (time x pca-decomposed confounds)
                # - run_idx: 1D array (time)
                # - bookkeeping stuff, e.g., sub (01), ses (1), task (face), work_dir, etc


if __name__ == '__main__':

    main()