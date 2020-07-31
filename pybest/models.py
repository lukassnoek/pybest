import numpy as np
from sklearn.metrics import r2_score


def cross_val_r2(estimator, X, Y, cv, groups=None, return_preds=False):
    """ Returns the average cross-validated R2-score, similar
    to sklearn's cross_val_score (which doesn't work with multioutput='raw_values').

    Parameters
    ----------
    estimator : sklearn estimator
        Scikit-learn compatible estimator (Regressor)
    X : array-like
        Two-dimensional predictor array
    Y : array-like
        Two-dimensional target array
    cv : sklearn cross-validation generator
        Scikit-learn compatible cross-validation generator
    """

    if return_preds:
        preds = np.zeros_like(Y)

    # Pre-allocate R2 array
    r2 = np.zeros(Y.shape[1])
    for i, (train_idx, test_idx) in enumerate(cv.split(X, Y, groups=groups)):
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        estimator.fit(X_train, Y_train)
        if return_preds:
            preds[test_idx, :] = estimator.predict(X_test)
            r2 += r2_score(Y_test, preds[test_idx, :], multioutput='raw_values')
        else:
            r2 += r2_score(Y_test, estimator.predict(X_test), multioutput='raw_values')
        
    # Average R2-scores across splits
    r2 /= cv.get_n_splits(groups=groups)
    if return_preds:
        return r2, preds
    else:
        return r2

    