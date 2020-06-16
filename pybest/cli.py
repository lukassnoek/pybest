import os
import click
import os.path as op
import nibabel as nib
from glob import glob
from .logging import logger
from .utils import check_parameters, set_defaults
from .utils import find_exp_parameters, find_data
from .preproc import preprocess_funcs, preprocess_confs, preprocess_events
from .preproc import load_preproc_data
from .noise_model import run_noise_processing, load_denoising_data
from .signal_model import run_signal_processing
from .constants import HRF_MODELS


@click.command()
@click.argument('bids_dir')
@click.argument('out_dir', default=None, required=False)
@click.option('--fprep_dir', default=None, required=False)
@click.option('--ricor_dir', default=None, required=False)
@click.option('--subject', default=None, required=False)
@click.option('--start-from', type=click.Choice(['preproc', 'noiseproc', 'signalproc']), default='preproc', required=False)
@click.option('--session', default=None, required=False)
@click.option('--task', default=None, required=False)
@click.option('--space', default='T1w', show_default=True)
@click.option('--hemi', type=click.Choice(['L', 'R']), default='L', show_default=True)
@click.option('--gm-thresh', default=0, show_default=True)  # maybe use a "mask" option
@click.option('--slice-time-ref', type=click.FLOAT, default=0.5, show_default=True)
@click.option('--high-pass-type', type=click.Choice(['dct', 'savgol']), default='dct', show_default=True)
@click.option('--high-pass', default=0.01, show_default=True)
@click.option('--tr', default=None, type=click.FLOAT, show_default=True)
@click.option('--decomp', default='pca', type=click.Choice(['pca', 'ica']), show_default=True)
@click.option('--n-comps', default=100, type=click.INT, show_default=True)
@click.option('--cv-repeats', default=2, type=click.INT, show_default=True)
@click.option('--cv-splits', default=5, type=click.INT, show_default=True)
@click.option('--skip-signalproc', is_flag=True)
@click.option('--single-trial-id', default=None, type=click.STRING, show_default=True)
@click.option('--hrf-model', default='kay', type=click.Choice(HRF_MODELS), show_default=True)
@click.option('--single-trial-noise-model', default='ols', type=click.Choice(['ols', 'ar1']), show_default=True)
@click.option('--regularize-hrf-model', is_flag=True)
@click.option('--single-trial-model', default='lsa', type=click.Choice(['lsa', 'lss']), show_default=True)
@click.option('--pattern-units', default='beta', type=click.Choice(['beta', 'zscore']), show_default=True)
@click.option('--uncorrelation', is_flag=True)
@click.option('--n-cpus', default=1, show_default=True)
@click.option('--save-all', is_flag=True)
def main(bids_dir, out_dir, fprep_dir, ricor_dir, subject, start_from, session, task, space, hemi,
         gm_thresh, slice_time_ref, high_pass_type, high_pass, tr, decomp, n_comps, cv_repeats, cv_splits,
         skip_signalproc, single_trial_id, hrf_model, single_trial_noise_model, regularize_hrf_model, single_trial_model, pattern_units, uncorrelation,
         n_cpus, save_all):
    """ Main API of pybest. """

    ##### set + check parameters #####
    cfg = locals()
    cfg = set_defaults(cfg, logger)
    check_parameters(cfg, logger)

    ##### find data #####
    cfg = find_exp_parameters(cfg, logger)

    ##### <start processing loop> #####
    for i, sub in enumerate(cfg['subject']):

        for ii, ses in enumerate(cfg['session'][i]):

            for task in cfg['task'][i][ii]:
                if task is None:
                    continue  # no data for this task in this run

                to_add = '' if ses is None else f' ses-{ses} '
                logger.info(f"Starting process for sub-{sub},{to_add} task-{task}")
                
                # Some bookkeeping
                if ses is None:
                    cfg['f_base'] = f"sub-{sub}_task-{task}"
                    cfg['save_dir'] = op.join(cfg['out_dir'], f'sub-{sub}')
                else:
                    cfg['f_base'] = f"sub-{sub}_ses-{ses}_task-{task}"
                    cfg['save_dir'] = op.join(cfg['out_dir'], f'sub-{sub}', f'ses-{ses}')
                
                if not op.isdir(cfg['out_dir']):
                    os.makedirs(cfg['out_dir'])

                for key, value in [('sub', sub), ('ses', ses), ('task', task)]:
                    # c_ stands for "current" (c_task = current task)
                    cfg['c_' + key] = value

                # ddict = data dictionary
                ddict = find_data(cfg, logger)
                
                # Start from preproc
                if start_from not in ['noiseproc', 'signalproc']:
                    ddict = preprocess_funcs(ddict, cfg, logger)
                    ddict = preprocess_confs(ddict, cfg, logger)
                    if not cfg['skip_signalproc']:
                        ddict = preprocess_events(ddict, cfg, logger)
                    else:
                        ddict['preproc_events'] = None
 
                # If we did preprocessing already ...
                if start_from == 'noiseproc':
                    ddict = load_preproc_data(ddict, cfg)
                
                # ... and didn't do noise processing yet ...
                if not start_from == 'signalproc':
                    # Run noise processing
                    ddict = run_noise_processing(ddict, cfg, logger)
                else:
                    # If we did do noise processing, load the denoising parameters
                    logger.info("Loading denoising parameters")
                    ddict = load_denoising_data(ddict, cfg)

                if not cfg['skip_signalproc']:
                    run_signal_processing(ddict, cfg, logger)


if __name__ == '__main__':
    main()
