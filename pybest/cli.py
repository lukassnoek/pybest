import os
import click
import os.path as op
import nibabel as nib
from glob import glob
from .utils import logger, set_defaults, find_exp_parameters, find_data
from .preproc import preprocess_func, preprocess_conf
from .preproc import load_preproc_data, save_preproc_data


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
@click.option('--gm-thresh', default=0.9, show_default=True)
@click.option('--high-pass-type', type=click.Choice(['dct', 'savgol']), default='dct', show_default=True)
@click.option('--high-pass', default=0.1, show_default=True)
@click.option('--savgol-order', default=4, show_default=True)
@click.option('--hemi', type=click.Choice(['L', 'R']), default='L', show_default=True)
@click.option('--tr', default=None, show_default=True)
@click.option('--nthreads', default=1, show_default=True)
def main(bids_dir, out_dir, fprep_dir, ricor_dir, subject, work_dir, start_from,
         session, task, space, gm_thresh, high_pass_type, high_pass, savgol_order, hemi, tr, nthreads):
    """ Main API of pybest. """

    ##### set defaults #####
    bids_dir, out_dir, fprep_dir, ricor_dir, work_dir, subject = set_defaults(
        bids_dir, out_dir, fprep_dir, ricor_dir, work_dir, subject, logger
    )

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
                    ##### <preprocessing> #####
                    func_data, run_idx = preprocess_func(
                        funcs,
                        gm_prob,
                        space,
                        logger,
                        high_pass_type,
                        high_pass,
                        savgol_order,
                        gm_thresh,
                        tr
                    )
                    
                    conf_data = preprocess_conf(
                        confs,
                        ricors,
                        high_pass_type,
                        high_pass,
                        savgol_order,
                        tr
                    )
                    logger.info("Finished preprocessing")
                    #save_preproc_data(sub, ses, task, func_data, conf_data, event_data, work_dir)
                elif start_from == 'noiseproc':
                    #func_data, run_idx = load_preproc_data(sub, ses, task, work_dir)
                    pass
                #elif start_from == 'signalproc':
                #    func_data, run_idx = load_denoised_data()
                #else:
                #    raise ValueError("Parameter '--start-from' should be 'preproc', 'noiseproc' or 'signalproc'!")

if __name__ == '__main__':

    main()