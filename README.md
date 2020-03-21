# pybest
PYthon package for Beta ESTimation (of single-trial fMRI data)

## Notes
[Notes about the project in a Google doc](https://docs.google.com/document/d/e/2PACX-1vQ1xuPuqeO6V-qccE1dHPkj53yHSVXbldqQMmLNqQt4HdlAQJljTrv7fEqw3WsDhF6dy63KG3tpWCtY/pub).

### Download example/test data
In the root of the directory, run:

```
./scripts/download_data
```

This will download one Fmriprep-preprocessed subject with one session (with 6 runs), which includes preprocessed volume and surface (gifti) data, confounds, and events. Note, it's 2.5 GB, so it might take a while.

### Installing
To install, clone the repository and run `pip install -e .` (the `-e` flag will install a development version).

### Using pybest
First, download the data. Then, check its API by:

```
pybest --help

Usage: pybest [OPTIONS] BIDS_DIR [OUT_DIR] [FPREP_DIR] [RICOR_DIR]

  Main API of pybest.

Options:
  --participant-label TEXT
  --session TEXT
  --task TEXT
  --space TEXT
  --high-pass FLOAT
  --hemi TEXT
  --tr FLOAT
  --help                    Show this message and exit.
```

After downloading the example data, you can test `pybest` as follows:

```
pybest pybest/data/ni-edu --space {T1w,fsaverage6} --tr 0.7
```

The `--tr` parameter is only necessary when using `fsaverage6` space (because the gifti header does not store the scan's TR).
