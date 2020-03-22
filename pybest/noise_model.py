import os.path as op
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from nilearn import masking
from .utils import tqdm_out


def run_noise_processing(func_data, conf_data, run_idx, mask, work_dir):
    
    scaler = StandardScaler()
    model = RidgeCV()

    n_run = np.unique(run_idx).size
    cv = GroupKFold(n_splits=n_run).split(conf_data, func_data[:, 0], groups=run_idx)
    
    r2_scores = np.zeros(func_data.shape[1])
    for train_idx, test_idx in tqdm(cv, file=tqdm_out):
        
        y_train = scaler.fit_transform(func_data[train_idx, :])
        y_test = scaler.fit_transform(func_data[test_idx, :])

        X_train = scaler.fit_transform(conf_data.iloc[train_idx, :])
        X_test = scaler.fit_transform(conf_data.iloc[test_idx, :])

        model.fit(X_train, y_train)
        # Overfitting to check
        y_pred = model.predict(X_train)
        r2_scores += r2_score(y_train, y_pred, multioutput='raw_values')
        
    r2_scores /= n_run
    if mask is not None:
        r2_scores = masking.unmask(r2_scores, mask)
        r2_scores.to_filename(op.join(work_dir, 'r2.nii.gz'))
    else:
        np.save('r2.npy', r2_scores)

