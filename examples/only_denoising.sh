# To skip "signalproc" (single-trial estimation), just don't set --single-trial-id or set --skip-signalproc
pybest ${bids_dir} \         # e.g. $PWD
	--subject ${sub_id} \    # e.g. 01
	--task expressive \      # name of task
    --space T1w \            # alternative: fsaverage, fsnative, MNI152NLin2009cAsym etc.
	--high-pass-type dct \   # alternative: savgol
    --high-pass 0.1 \        # high-pass cutoff in Hz
    --tr 1.7 \               # if not provided, read from nifti header
    --gm-thresh 0 \          # gray-matter mask threshold (0 = use all voxels within *desc-brain_mask.nii.gz)
	--n-comps 50 \           # number of noise components to try out
	--cv-splits 5 \          # number of folds ("splits") in noise CV
    --cv-repeats 1 \         # number of noise CV repeats  
	--n-cpus 5 \             # number of cpus used (runs are parallelized)
	--save-all               # whether to save extra information to disk