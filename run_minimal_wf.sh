#!/bin/sh

RESULTFOLDER=results
WORKFOLDER=results/_work

pyprophet-cli prepare --data-folder tests/data --data-filename-pattern test_data.txt --work-folder $WORKFOLDER

pyprophet-cli subsample --random-seed 43 --data-folder tests/data --data-filename-pattern test_data.txt --work-folder $WORKFOLDER --sample-factor 0.5

pyprophet-cli learn --random-seed 43 --work-folder $WORKFOLDER --ignore-invalid-scores

pyprophet-cli apply_weights --work-folder $WORKFOLDER --data-folder tests/data/

pyprophet-cli score --work-folder $WORKFOLDER --data-folder tests/data/ --overwrite-results  --d-score-cutoff -100000 --result-folder $RESULTFOLDER
