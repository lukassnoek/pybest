import matplotlib
matplotlib.use('Agg')

import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import image, masking
from nistats.first_level_model import FirstLevelModel
from nistats.design_matrix import _cosine_drift, make_first_level_design_matrix
from sklearn.metrics import pairwise_distances


def fit_flm(func, events, mask, out='preproc'):
    tr = 1.317#func.header['pixdim'][4]
    nvol = func.shape[-1]
    ft = np.linspace(tr / 2, nvol * tr + tr / 2, nvol, endpoint=False)
    flm = FirstLevelModel(
        t_r=tr,
        hrf_model='glover + dispersion + derivative',
        drift_model=None,
        high_pass=None,
        mask_img=mask,
        smoothing_fwhm=None,
        noise_model='ols',
        standardize=True,
        signal_scaling=False,
        minimize_memory=False
    )
    #events['trial_type'] = np.arange(events.shape[0])
    #events['trial_type'] = 'face'
    dm = make_first_level_design_matrix(
        frame_times=ft,
        events=events,
        hrf_model='glover',
        drift_model=None
    ).iloc[:, :-1]
    face = dm.sum(axis=1)
    dm = dm - dm.mean(axis=0)
    dm['face'] = face
    dm['icept'] = 1

    plt.figure(figsize=(15, 15))
    plt.imshow(dm, aspect='auto')
    plt.savefig('design.png')

    flm.fit(run_imgs=func, design_matrices=dm)
    flm.compute_contrast('face').to_filename(f'face-{out}.nii.gz')
    #flm.compute_contrast('icept').to_filename(f'icept-{out}.nii.gz')
    #flm.predicted[0].to_filename('pred.nii.gz')
    flm.r_square[0].to_filename(f'r2-{out}.nii.gz')
    cons = []
    to_loop = events['trial_type']
    for tt in to_loop:
        cons.append(flm.compute_contrast(tt))

    imgs = image.concat_imgs(cons)
    imgs.to_filename(f'pattern-{out}.nii.gz')
    """
    #tmp = masking.apply_mask(imgs, mask)
    #dm = np.array([-1 if s.split('_')[1] == 'neutral' else 1 for s in sorted(to_loop, key=lambda x: x.split('_')[1])])[:, np.newaxis]
    #b = np.linalg.inv(dm.T @ dm) @ dm.T @ tmp
    #varb = np.sqrt(np.sum((tmp - dm @ b)**2, axis=0)) / 39
    #b = b / varb
    #masking.unmask(b, mask).to_filename('smi.nii.gz')
    """
data_dir = 'pybest/data/FEED/derivatives/pybest/work/sub-05/ses-face1'
events = data_dir + '/preproc/sub-05_ses-face1_task-expressive_desc-preproc_events.tsv'
events = pd.read_csv(events, sep='\t')

func = data_dir + '/denoising/sub-05_ses-face1_task-expressive_desc-denoised_bold.nii.gz'
#func = data_dir + '/preproc/sub-05_ses-face1_task-expressive_desc-preproc_bold.nii.gz'

func = nib.load(func)
mask = data_dir + '/preproc/sub-05_ses-face1_task-expressive_desc-preproc_mask.nii.gz'
fit_flm(func, events, mask, out='denoised')