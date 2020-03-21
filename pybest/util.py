import logging
import numpy as np
import nibabel as nib
from tqdm import tqdm
from nilearn import image, masking, signal
from nistats.design_matrix import _cosine_drift as dct_set


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-8s] [%(levelname)-7.7s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('pybest')


def _load_gifti(f):
    """ Load gifti array. """
    f_gif = nib.load(f)
    return np.vstack([arr.data for arr in f_gif.darrays])


def preprocess(funcs, mask, space, logger, high_pass=0.1, gm_thresh=0.9, tr=.7):

    if 'fs' not in space:
        if mask is None:  # create fmask intersection
            logger.info("Creating mask by intersection of function masks")
            fmasks = [f.replace('desc-preproc_bold', 'desc-brain_mask') for f in funcs]
            mask = masking.intersect_masks(fmasks, threshold=0.8)
        else:
            logger.info("Creating mask using GM probability map")
            mask = image.resample_to_img(mask, funcs[0])
            mask_data = mask.get_fdata()
            mask_data = (mask_data >= gm_thresh).astype(int)
            mask = nib.Nifti1Image(mask_data, affine=mask.affine)
    else:
        mask = None

    logger.info("Starting preprocessing of functional data ...")
    all_run_data = []
    run_idx = []
    for i, func in enumerate(tqdm(funcs)):

        if 'fs' in space:  # assume gifti
            data = _load_gifti(func)
        else:
            data = masking.apply_mask(func, mask)
            hdr = nib.load(func).header
            tr = hdr['pixdim'][4]

        n_vol = data.shape[0]
        frame_times = np.linspace(0.5 * tr, n_vol * (tr + 0.5), n_vol, endpoint=False)    
        hp_set = dct_set(high_pass, frame_times)[:, :-1]  # remove intercept
        
        data = signal.clean(data, detrend=True, standardize=True, confounds=hp_set)
        all_run_data.append(data)

        run_idx.append(np.ones(n_vol) * i)

    all_run_data = np.vstack(all_run_data)
    run_idx = np.concatenate(run_idx)

    logging.info(
        f"HP-filtered/normalized data has {all_run_data.shape[0]} timepoints "
        f"(across {len(funcs)} runs) and {all_run_data.shape[1]} voxels"
    )

    return all_run_data, run_idx    