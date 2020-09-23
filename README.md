# pybest
PYthon package for Beta ESTimation (of single-trial data). Also does some nifty denoising.

## Warning
This package is still in development and its API might change. Also, note that it has a very extensive API (many options), but each option has a sensible default.

## What you need to run pybest ...
* Fmriprep-preprocessed data (or data with *desc-preproc_bold, *desc-brain_mask, and *desc-confounds_regressors.tsv files)
* Tsv-formatted (BIDS-style) events-files (not necessary if you want to do denoising only)
* Lots of RAM (especially when using multiple CPUs)

### Installing
To install, clone the repository and run `pip install -e .` (the `-e` flag will install a development version, which you can omit). Note: you need to install the master version of `nilearn` from Github yourself (clone + pip install .), because `pybest` uses the latest (not-yet-released) version of `nilearn`.

### Using pybest
The API is relatively well documented. Check it out by running:

```
pybest --help
```

The only actual obligatory argument is the path to your Fmriprep directory. The `--subject`, `--session`, and `--task` can be used to restrict `pybest` to process only a part of your dataset. These parameters are optional, but not setting them will cause `pybest` to process *everything* it can find in your Fmriprep directory. 

The other parameters all have sensible defaults (but which you can change as desired). By default, `pybest` will do the following:
1. Preprocess the functional data (high-pass at 0.01 Hz + standardization) and confounds (high-pass + PCA to extract 50 components);
2. Perform "within-run" style noise-processing using 5 fold cross-validation and 1-50 noise components for each run separately (no `--regularize-n-comps`);
3. Run a least-squares-all (LSA) style using the Glover HRF on the denoised data, but only if `--single-trial-id {identifier}` is set (e.g. `--single-trial-id face_`). All events with this identifier in the `trial_type` column in the associated events-file will be treated as a separate trial. All other events are modelled as a condition. The "patterns" are whitened with the design covariance and saved as "betas" (alternative: zscore).
