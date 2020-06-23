import os
import click
import os.path as op
import nibabel as nib
from glob import glob
from .logging import get_logger
from .constants import HRF_MODELS
from .bookkeeping import check_parameters, set_defaults, find_exp_parameters, find_data
from .preproc import preprocess_funcs, preprocess_confs_fmriprep, preprocess_confs_noise_pool
from .preproc import preprocess_events, load_preproc_data
from .noise_model import run_noise_processing, load_denoising_data
from .signal_model import run_signal_processing


@click.command()
# 1. Positional args
@click.argument('bids_dir')
@click.argument('out_dir', default=None, required=False)
# 2. Which data should we select?
@click.option('--start-from', type=click.Choice(['preproc', 'noiseproc', 'signalproc']), default='preproc', required=False)
@click.option('--fprep-dir', default=None, required=False)
@click.option('--ricor-dir', default=None, required=False)
@click.option('--subject', default=None, required=False)
@click.option('--session', default=None, required=False)
@click.option('--ignore-sessions', is_flag=True, required=False)
@click.option('--task', default=None, required=False)
@click.option('--space', default='T1w', show_default=True)
@click.option('--hemi', type=click.Choice(['L', 'R']), default='L', show_default=True)
# 3. Preproc options
@click.option('--gm-thresh', default=0., show_default=True)  # maybe use a "mask" option
@click.option('--slice-time-ref', type=click.FLOAT, default=0.5, show_default=True)
@click.option('--high-pass-type', type=click.Choice(['dct', 'savgol']), default='dct', show_default=True)
@click.option('--high-pass', default=0.01, show_default=True)
# 4. Noiseproc options
@click.option('--skip-noiseproc', is_flag=True)
@click.option('--noise-source', default='fmriprep', type=click.Choice(['fmriprep', 'noisepool']))
@click.option('--decomp', default='pca', type=click.Choice(['pca', 'ica']), show_default=True)
@click.option('--n-comps', default=100, type=click.INT, show_default=True)
@click.option('--cv-repeats', default=1, type=click.INT, show_default=True)
@click.option('--cv-splits', default=5, type=click.INT, show_default=True)
# 5. Signalproc options
@click.option('--skip-signalproc', is_flag=True)
@click.option('--signalproc-type', default='single-trial', type=click.Choice(['single-trial', 'glmdenoise']))
@click.option('--contrast', default=None, type=click.STRING, show_default=True)
@click.option('--bootstraps', default=100, type=click.INT, show_default=True)
# 5.1. Single-trial options
@click.option('--single-trial-id', default=None, type=click.STRING, show_default=True)
@click.option('--hrf-model', default='glover', type=click.Choice(HRF_MODELS), show_default=True)
@click.option('--single-trial-noise-model', default='ols', type=click.Choice(['ols', 'ar1']), show_default=True)
@click.option('--regularize-hrf-model', is_flag=True)
@click.option('--single-trial-model', default='lsa', type=click.Choice(['lsa', 'lss']), show_default=True)
@click.option('--pattern-units', default='beta', type=click.Choice(['beta', 'zscore']), show_default=True)
@click.option('--uncorrelation', is_flag=True)
# 6. Misc options
@click.option('--smoothing-fwhm', default=None, type=click.FLOAT, show_default=True)
@click.option('--n-cpus', default=1, show_default=True)
@click.option('--save-all', is_flag=True)
@click.option('--verbose', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), show_default=True)
def main(bids_dir, out_dir, start_from, fprep_dir, ricor_dir, subject, session, ignore_sessions, task, space, hemi,
         gm_thresh, slice_time_ref, high_pass_type, high_pass, skip_noiseproc, noise_source, decomp, n_comps, cv_repeats, cv_splits, skip_signalproc,
         signalproc_type, contrast, bootstraps, single_trial_id, hrf_model, single_trial_noise_model, regularize_hrf_model, single_trial_model,
         pattern_units, uncorrelation, smoothing_fwhm, n_cpus, save_all, verbose):
    """ Main API of pybest. """
    
    ##### set + check parameters #####
    cfg = locals()
    logger = get_logger(verbose)
    cfg = set_defaults(cfg, logger)
    check_parameters(cfg, logger)

    ##### find data #####
    cfg = find_exp_parameters(cfg, logger)

    ##### <start processing loop> #####
    # Loop over subjects, sessions, tasks
    for i, sub in enumerate(cfg['subject']):
        for ii, ses in enumerate(cfg['session'][i]):

            for task in cfg['task'][i][ii]:
                
                if task is None:
                    # no data for this task in this session, so skip
                    continue

                s = '' if ses is None else f' ses-{ses},'
                logger.info(f"Starting process for sub-{sub},{s} task-{task}")
                
                # Some bookkeeping
                space_idf = f'{space}_hemi-{hemi}' if 'fs' in space else space

                # f_base: base for output files
                if ses is None:  # no separate session output dir
                    cfg['f_base'] = f"sub-{sub}_task-{task}_space-{space_idf}"
                    cfg['save_dir'] = op.join(cfg['out_dir'], f'sub-{sub}')
                else:
                    cfg['f_base'] = f"sub-{sub}_ses-{ses}_task-{task}_space-{space_idf}"
                    cfg['save_dir'] = op.join(cfg['out_dir'], f'sub-{sub}', f'ses-{ses}')
                
                for key, value in [('sub', sub), ('ses', ses), ('task', task)]:
                    # c_ stands for "current" (c_task = current task)
                    cfg['c_' + key] = value

                # ddict = data dictionary
                ddict = find_data(cfg, logger)
                
                # Start from preproc
                if start_from not in ['noiseproc', 'signalproc']:
                    # Preprocess funcs, confs, events
                    ddict = preprocess_funcs(ddict, cfg, logger)
                    if not cfg['skip_signalproc']:  # ignore event files
                        ddict = preprocess_events(ddict, cfg, logger)
                    else:
                        ddict['preproc_events'] = None

                    if cfg['noise_source'] == 'fmriprep':  # use fmriprep confound files
                        ddict = preprocess_confs_fmriprep(ddict, cfg, logger)
                    else:  # estimate confounds from GLMdenoise-style noise pool 
                        ddict = preprocess_confs_noise_pool(ddict, cfg, logger)
                
                # If we did preprocessing already ...
                if start_from == 'noiseproc':
                    ddict = load_preproc_data(ddict, cfg)
                
                # ... and didn't do noise processing yet ...
                if not start_from == 'signalproc':
                    ddict = run_noise_processing(ddict, cfg, logger)
                else:
                    # If we did do noise processing, load the denoising parameters
                    logger.info("Loading denoising parameters")
                    ddict = load_denoising_data(ddict, cfg)

                run_signal_processing(ddict, cfg, logger)


if __name__ == '__main__':
    main()
