import os
import os.path as op
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from nilearn import signal, masking
from sklearn.linear_model import Ridge
from nistats.first_level_model import run_glm
from nistats.design_matrix import make_first_level_design_matrix
from nistats.reporting import plot_design_matrix
from nistats.contrasts import compute_contrast
from scipy.linalg import sqrtm
from .preproc import hp_filter
from .utils import get_run_data, yield_uniq_params


here = op.dirname(__file__)
HRFS = pd.read_csv(op.join(here, 'data', 'hrf_ts.tsv'), sep='\t', index_col=0)

def _run_parallel(run, out_dir, ddict, cfg, logger):
    tr = ddict['tr']
    Y, conf, events = get_run_data(ddict, run, func_type='denoised')
    to_subtract = (ddict['run_idx'] < run).sum() * tr
    events['trial_type'] = [f'trial_{str(i).zfill(3)}' for i in range(events.shape[0])]
    events['onset'] -= to_subtract
    n_vol = Y.shape[0]
    st_ref = cfg['slice_time_ref']
    ft = np.linspace(st_ref * tr, n_vol * tr + st_ref * tr, n_vol, endpoint=False)
    
    # Make single-trial (LSA) design matrix
    create_design_matrix(0, ddict['tr'], ft, events)
    exit()
    X = make_first_level_design_matrix(
        frame_times=ft,
        events=events,
        hrf_model='glover',
        drift_model=None
    ).iloc[:, :-1]  # remove constant        
    n_pred = X.shape[1]

    # Fit intercept only model
    X['intercept'] = X.sum(axis=1)
    X_icp = X.iloc[:, -1:].copy()  # remove single trials
    # Orthogonalize high-pass filter to intercept model (also normalizes)
    X_icp.iloc[:, :] = hp_filter(X_icp.to_numpy(), ddict, cfg, logger)
    labels, results = run_glm(Y, X_icp.to_numpy(), noise_model='ols')

    # Store average response for inspection
    icept_z = compute_contrast(labels, results, con_val=[1], contrast_type='t').z_score()
    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-intercept_zscore.nii.gz')
    masking.unmask(icept_z, ddict['mask']).to_filename(f_out)

    # Fit trial model on residuals
    Y_resids = get_param_from_glm('residuals', labels, results, X_icp, time_series=True)
    X_trial = X.iloc[:, :n_pred].copy()
    labels, results = run_glm(Y_resids, X_trial.to_numpy(), noise_model='ols')
    
    #resids = get_param_from_glm('residuals', labels, results, X_trial, time_series=True)
    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-trial_residuals.nii.gz')
    masking.unmask(Y_resids, ddict['mask']).to_filename(f_out)
    
    trial_patterns = np.zeros((n_pred, Y_resids.shape[1]))
    for i, col in enumerate(X_trial.columns[:n_pred]):
        cvec = np.zeros(X_trial.shape[1])
        cvec[X_trial.columns.tolist().index(col)] = 1
        trial_patterns[i, :] = compute_contrast(labels, results, con_val=cvec, contrast_type='t').effect_size()

    #D = sqrtm(np.linalg.inv(np.cov(X_trial.to_numpy().T)))
    #trial_patterns = D @ trial_patterns
    
    f_out = op.join(out_dir, cfg['f_base'] + f'_run-{run+1}_desc-trial_beta.nii.gz')
    masking.unmask(trial_patterns, ddict['mask']).to_filename(f_out)



def run_signal_processing(ddict, cfg, logger):
    """ Runs signal processing. """
    
    logger.info(f"Starting signal analysis.")

    sub, ses, task = cfg['sub'], cfg['ses'], cfg['task']
    out_dir = op.join(cfg['work_dir'], f'sub-{sub}', f'ses-{ses}', 'best')
    if not op.isdir(out_dir):
        os.makedirs(out_dir)
        
    out = Parallel(n_jobs=cfg['n_cpus'])(delayed(_run_parallel)
        (run, out_dir, ddict, cfg, logger) for run in tqdm(np.unique(ddict['run_idx']))
    )


def get_param_from_glm(name, labels, results, dm, time_series=False):
    if time_series:
        data = np.zeros((dm.shape[0], labels.size))
    else:
        data = np.zeros_like(labels)

    for lab in np.unique(labels):
        data[..., labels == lab] = getattr(results[lab], name)
    
    return data



def create_design_matrix(hrf_idx, tr, frame_times, events):
    from nistats.experimental_paradigm import check_events
    from nistats.hemodynamic_models import _sample_condition, _resample_regressor
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d, pchip

    ovs = 10

    t_hrf = HRFS.index.copy()
    hrf = HRFS.iloc[:, hrf_idx].to_numpy()
    f = interp1d(t_hrf, hrf)
    t_high = np.linspace(0, 50, num=hrf.size*ovs, endpoint=True) 
    hrf = f(t_high).T
    np.set_printoptions(suppress=True)

    osf = tr / (0.1 / ovs)
    trial_type, onset, duration, modulation = check_events(events)
    X = np.zeros((frame_times.size, np.unique(trial_type).size))
    for i, condition in enumerate(np.unique(trial_type)):
        condition_mask = (trial_type == condition)
        exp_condition = (onset[condition_mask],
                         duration[condition_mask],
                         modulation[condition_mask])
        hr_regressor, hr_frame_times = _sample_condition(exp_condition, frame_times, osf, 0)
        conv_reg = np.convolve(hr_regressor, hrf)[:hr_regressor.size]
        plt.plot(conv_reg)
        #f = interp1d(hr_frame_times, conv_reg)
        #X[:, i] = f(frame_times).T
    
    plt.savefig('test.png')


