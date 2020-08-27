import os
import click
import os.path as op
import nibabel as nib
from glob import glob

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np

from .logging import get_logger
from .constants import HRF_MODELS
from .bookkeeping import check_parameters, set_defaults, find_exp_parameters, find_data
from .preproc import preprocess_funcs, preprocess_confs_fmriprep, preprocess_confs_noise_pool
from .preproc import preprocess_events, load_preproc_data
from .noise_model import run_noise_processing, load_denoising_data
from .signal_model import run_signal_processing


@click.command()
# 1. Positional args
@click.argument('fprep_dir')  # Path to Fmriprep output directory
@click.argument('bids_dir', default=None, required=False)  # Optional BIDS-directory (for event-files)
# 2. Which data should we select?
@click.option('--out-dir', default=None, help='Output directory (default: same level as Fmriprep directory)')
@click.option('--start-from', type=click.Choice(['preproc', 'noiseproc', 'signalproc']), default='preproc', help='Stage to start analysis from')
@click.option('--ricor-dir', default=None, help='RETROICOR/physiology directory (default: None, assumed to not exist)')
@click.option('--subject', default=None, help='Subject identifier (e.g., 01); default: all subjects pybest can find')
@click.option('--session', default=None, help='Session identifier (e.g., mapper1); default: all sessions pybest can find')
@click.option('--pool-sessions', is_flag=True, help='Flag: whether to pool data (from a given task) across sessions; helpful for GLMdenoise')
@click.option('--task', default=None, help='Task identifier (e.g., prf); default: all tasks pybest can find')
@click.option('--space', default='T1w', show_default=True, help='Output space of data to be processed (e.g., T1w, MNI152NLin2009cAsym)')
@click.option('--hemi', type=click.Choice(['L', 'R']), default='L', show_default=True, help='Hemisphere to process (only relevant when dealing with surface space data)')
# 3. Preproc options
@click.option('--gm-thresh', default=0., show_default=True, help='Threshold for gray-matter mask (if 0, Fmriprep brain masks are used)')  # maybe use a "mask" option
@click.option('--slice-time-ref', type=click.FLOAT, default=0.5, show_default=True, help='Slice to adjust event onsets to (depends on slice-time correction)')
@click.option('--high-pass-type', type=click.Choice(['dct', 'savgol']), default='dct', show_default=True, help='Type high-pass filter')
@click.option('--high-pass', default=0.01, show_default=True, help='High-pass cutoff in Hz')
@click.option('--trial-filter', type=click.STRING, help='String to pass to DataFrame.query to filter trials')
# 4. Noiseproc options
@click.option('--skip-noiseproc', is_flag=True, help='Flag: whether to skip noiseprocessing (mainly used to testing/debugging)')
@click.option('--noise-source', default='fmriprep', type=click.Choice(['fmriprep', 'noisepool']), help='Source of noise predictor (fmriprep: confound.tsv files, noisepool: GLMdenoise style noise pool estimation)')
@click.option('--decomp', default='pca', type=click.Choice(['pca', 'ica']), show_default=True, help='Decomposition algorithm used for noise predictor set')
@click.option('--n-comps', default=50, type=click.INT, show_default=True, help='Number of noise components to evaluate')
@click.option('--noiseproc-type', default='within', type=click.Choice(['within', 'between']), show_default=True, help='Type of noise processing (within: within-run noise cross-validation, between: across-run GLMdenoise-style signal cross-validation)')
@click.option('--cv-repeats', default=1, type=click.INT, show_default=True, help='Number of cross-validation repeats in noise processing (only relevant when noiseproc-type is "within")')
@click.option('--cv-splits', default=5, type=click.INT, show_default=True, help='Number of cross-validation splits in noise processing (folds; only relevant when noiseproc-type is "within"')
@click.option('--regularize-n-comps', is_flag=True, help='Flag: whether to "regularize" number of noise components by picking the same number across runs')
@click.option('--argmax-percent', default=5., type=click.FLOAT, help='How much (in percent) the argmax r2 may deviate from the the max')
# 5. Signalproc options
@click.option('--skip-signalproc', is_flag=True, help='Flag: whether to skip signal processing (only noiseproc)')
@click.option('--signalproc-type', default='single-trial', type=click.Choice(['single-trial', 'glmdenoise']), help='Type of signal processing (single-trial: LSA/LSS style model, glmdenoise: cross-validated model)')
@click.option('--contrast', default=None, type=click.STRING, show_default=True, help='Contrast to evaluate (e.g., "4*face - object - place - character - body"); use double quotes!')
# 5.1. Single-trial options
@click.option('--single-trial-id', default=None, type=click.STRING, show_default=True, help='Identifier for single-trial events (e.g., face_)')
@click.option('--hrf-model', default='glover', type=click.Choice(HRF_MODELS), show_default=True, help='HRF type to use for anything involving HRF convolution')
@click.option('--single-trial-noise-model', default='ols', type=click.Choice(['ols', 'ar1']), show_default=True, help='Noise model for GLM estimation')
@click.option('--regularize-hrf-model', is_flag=True, help='Flag: whether to regularize the HRF model (see --regularize-n-comps)')
@click.option('--single-trial-model', default='lsa', type=click.Choice(['lsa', 'lss']), show_default=True, help='What type of single-trial model to estimate')
@click.option('--pattern-units', default='beta', type=click.Choice(['beta', 'zscore']), show_default=True, help='What type of stat images to compute/save')
@click.option('--uncorrelation', is_flag=True, help='Whether to prewhiten the patterns (R) with the design correlations: sqrtm(np.cov(X.T)) @ R (see Soch et al., 2020, NeuroImage)')
# 6. Misc options
@click.option('--smoothing-fwhm', default=None, type=click.FLOAT, help='Whether to smooth statistic maps (default: no smoothing)')
@click.option('--n-cpus', default=1, show_default=True, help='Number of CPUs to use (mainly to process runs in parallel)')
@click.option('--save-all', is_flag=True, help='Save more outputs than strictly necessary (but nice to inspect your data in detail)')
@click.option('--save-mgz', is_flag=True, help='Save surface files as mgz instead of npy files')
@click.option('--verbose', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), show_default=True, help='Verbosity level')
def main(fprep_dir, bids_dir, out_dir, start_from, ricor_dir, subject, session, pool_sessions, task, space, hemi,
         gm_thresh, slice_time_ref, high_pass_type, high_pass, trial_filter, skip_noiseproc, noise_source, decomp, n_comps,
         noiseproc_type, cv_repeats, cv_splits, regularize_n_comps, argmax_percent, skip_signalproc, signalproc_type, contrast,
         single_trial_id, hrf_model, single_trial_noise_model, regularize_hrf_model, single_trial_model, pattern_units, uncorrelation,
         smoothing_fwhm, n_cpus, save_all, save_mgz, verbose):
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
                    logger.warn(f"No data for sub-{sub}, ses-{ses} - skipping ...")
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
