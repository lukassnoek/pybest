# pybest
PYthon package for Beta ESTimation (of single-trial fMRI data)

## Ideas
Create a package that does single-trial ("beta") pattern estimation using regularized models (ridge) and state-of-the-art denoising and custom HRF fitting (a la Kendrick Kay). 

### Global package structure:
* Custom scikit-learn (nilearn) style pipeline with scaling, denoising, and single-trial estimation
* Should be able to plug into sklearn's GridSearchCV (to optimize for n_components and lambda)
* Should be able to plug into KFold CV generator
* Should allow for some form of custom HRF integration 

### Features to implement / try out
* Create a new CV class that creates *continuous* KFold partitions (because autocorrelation)
* Pipeline: scaling, PCA, 
* Create ridge with AR1 correction to account for left-over autocorrelation? (bit complex and computationally heavy, because we need to fit an AR1 model for each voxel separately)
* Make an efficient vectorized version of multivoxel ridge: precompute inv(X.T @ X + lambdaI) @ X.T for many lambdas ("hat matrix"), then only need to do hat_matrix @ y
* Loop over Kendrick's list of HRF shapes in single trial estimation model
* If putting trial-regressors and noise regressors in single model: banded ridge?

### Open questions
* Should we include our trial-regressors into the denoising matrix?
* Should we cross-validate PCA estimation?
* Should we high-pass data (savitsky-golay or DCT) before CV? (I think yes)
* Should we standardize (z-score) each run separately? (I think yes)
* Should we concatenate runs at some point? Maybe within session? This gives us many more datapoints to fit (but we need to assume that the true "noise models" are stationary across runs) and in that case we can do leave-one-run-out CV
* Should we, in general, fit the single-trial model separately from the denoising model? The single-trial model has to be cross-validated as well because we're probably going to use Kendrick's HRF set (which we'll loop over)
* Maybe do not cross-validate across sessions? (Because I think the noise sources/contributions are not stationary across sessions)

### Download example/test data
In the root of the directory, run:


```
./scripts/download_data
```

This will download one Fmriprep-preprocessed subject with one session (with 8 runs), which includes preprocessed volume and surface (gifti) data, confounds, and events. Note, it's 6.6 GB, so it might take a while.
