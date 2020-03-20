# pybest
PYthon package for Beta ESTimation (of single-trial fMRI data)

## Ideas
Create a package that does single-trial ("beta") pattern estimation using regularized models (ridge) and state-of-the-art denoising and custom HRF fitting (a la Kendrick Kay). 

### Global package structure:
* Custom scikit-learn (nilearn) style pipeline with scaling, denoising, and single-trial estimation
* Should be able to plug into sklearn's GridSearchCV (to optimize for n_components and lambda)
* Should be able to plug into KFold CV generator

### Features to implement / try out
* Create a new CV class that creates *continuous* KFold partitions (because autocorrelation)
* Pipeline: scaling, PCA, 
* Create ridge with AR1 correction to account for left-over autocorrelation? (bit complex and computationally heavy, because we need to fit an AR1 model for each voxel separately)

### Open questions
* Should we include our trial-regressors into the denoising matrix?
* Should we cross-validate PCA estimation?
* Should we high-pass data (savitsky-golay or DCT) before CV? (I think yes)
* Should we standardize (z-score) each run separately? (I think yes)
* Should we concatenate runs at some point? Maybe within session? This gives us many more datapoints to fit (but we need to assume that the true "noise models" are stationary across runs) and in that case we can do leave-one-run-out CV
