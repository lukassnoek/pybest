import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from nistats.first_level_model import run_glm
from nistats.contrasts import compute_contrast
from nilearn import image, masking

pybest_dir = 'pybest/data/ni-edu/derivatives/pybest/sub-02/ses-1'

mask = pybest_dir + '/preproc/sub-02_ses-1_task-face_desc-preproc_mask.nii.gz'
trials = sorted(glob(pybest_dir + '/best/*desc-trial*'))
for i in range(len(trials)):
    trials[i] = image.clean_img(trials[i], detrend=False, standardize=True)

Y = masking.apply_mask(image.concat_imgs(trials), mask)

events = pybest_dir + '/preproc/sub-02_ses-1_task-face_desc-preproc_events.tsv'
events = pd.read_csv(events, sep='\t').query("trial_type != 'rating' and trial_type != 'response'")
events.loc[:, 'face_eth'] = ['asian' if 'sian' in s else s for s in events['face_eth']]
events.loc[:, 'trial_type'] = [s[-7:] for s in events.loc[:, 'trial_type']]
X = events.loc[:, ['subject_dominance', 'subject_trustworthiness', 'subject_attractiveness']]
X /= X.mean(axis=0)
X = pd.concat((X, pd.get_dummies(events.loc[:, 'trial_type'])), axis=1)
X = pd.concat((X, pd.get_dummies(events.loc[:, 'face_eth'])), axis=1)
labels, results = run_glm(Y, X.to_numpy(), noise_model='ols')

for i in range(X.shape[1]):
    cvec = np.zeros(X.shape[1])
    cvec[i] = 1
    zscores = compute_contrast(labels, results, con_val=cvec, contrast_type='t').z_score()
    zscores = masking.unmask(zscores, mask)
    #zscores = image.smooth_img(zscores, fwhm=4)
    zscores.to_filename(f"{X.columns[i]}.nii.gz")


data = np.zeros_like(labels)

for lab in np.unique(labels):
    data[..., labels == lab] = getattr(results[lab], 'r_square')

masking.unmask(data, mask).to_filename('rsq.nii.gz')
