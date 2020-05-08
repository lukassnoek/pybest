import pandas as pd
import nibabel as nib
import numpy as np
from nilearn import image, masking
from nistats.first_level_model import FirstLevelModel
from nistats.design_matrix import _cosine_drift, make_first_level_design_matrix
from sklearn.metrics import pairwise_distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ffa_mask = '/home/lsnoek1/spinoza_data/ni-edu/bids/derivatives/floc/sub-02/rois/sub-02_task-flocBLOCKED_space-T1w_desc-face_zscore.nii.gz'
ffa_mask = image.math_img('(img > 8).astype(int)', img=ffa_mask)

data_dir = 'pybest/data/ni-edu/derivatives/pybest/work/sub-02/ses-1'
events = data_dir + '/preproc/sub-02_ses-1_task-face_run-1_desc-preproc_events.tsv'
events = pd.read_csv(events, sep='\t')
func = data_dir + '/denoising/sub-02_ses-1_task-face_run-1_desc-denoised_bold.nii.gz'
#func = 'pybest/data/ni-edu/derivatives/fmriprep/sub-02/ses-1/func/sub-02_ses-1_task-face_acq-Mb4Mm27Tr700_run-1_space-T1w_desc-preproc_bold.nii.gz'

func = nib.load(func)
ft = np.linspace(0.35, func.shape[-1]*0.7+0.35, func.shape[-1], endpoint=False)
#dct = _cosine_drift(0.01, ft)
#func = image.clean_img(func, detrend=False, standardize=True, high_pass=None, confounds=dct)

#func = image.clean_img(func, detrend=False, standardize=True, high_pass=0.02, t_r=0.7)

mask = data_dir + '/preproc/sub-02_ses-1_task-face_desc-preproc_mask.nii.gz'
#mask = 'pybest/data/ni-edu/derivatives/fmriprep/sub-02/ses-1/func/sub-02_ses-1_task-face_acq-Mb4Mm27Tr700_run-1_space-T1w_desc-brain_mask.nii.gz'

flm = FirstLevelModel(
    t_r=0.7,
    hrf_model='glover + derivative + dispersion',
    drift_model=None,
    high_pass=None,
    mask_img=mask,
    smoothing_fwhm=5,
    noise_model='ols'
)
dm = make_first_level_design_matrix(
    frame_times=ft,
    events=events,
    hrf_model='glover',
    drift_model=None
).iloc[:, :-1]
dm['face'] = dm.sum(axis=1)
dm = (dm - dm.mean(axis=0)) / dm.std(axis=0)
dm['icept'] = 1

flm.fit(run_imgs=func, design_matrices=dm)#events=events)
flm.compute_contrast('face').to_filename('face.nii.gz')
flm.compute_contrast('icept').to_filename('icept.nii.gz')

cons = []
to_loop = events['trial_type']
for tt in sorted(to_loop, key=lambda x: x.split('_')[1]):
    cons.append(flm.compute_contrast(tt))


imgs = image.concat_imgs(cons)
imgs.to_filename('test.nii.gz')
tmp = masking.apply_mask(imgs, mask)
dm = np.array([-1 if s.split('_')[1] == 'neutral' else 1 for s in sorted(to_loop, key=lambda x: x.split('_')[1])])[:, np.newaxis]
b = np.linalg.inv(dm.T @ dm) @ dm.T @ tmp
varb = np.sqrt(np.sum((tmp - dm @ b)**2, axis=0)) / 39
b = b / varb
masking.unmask(b, mask).to_filename('smi.nii.gz')

data = masking.apply_mask(imgs, ffa_mask)
rdm = 1-np.corrcoef(data)
plt.imshow(rdm)
plt.savefig('test.png')