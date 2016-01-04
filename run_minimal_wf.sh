#!/bin/sh

RESULTFOLDER=results
WORKFOLDER=results/_work

rm $WORKFOLDER/*

DATA_FOLDER=$HOME/Downloads
PATTERN="g*.tsv"

DATA_FOLDER=./tests/data
PATTERN="*.txt"

pyprophet-cli prepare --data-folder $DATA_FOLDER --data-filename-pattern "$PATTERN" --work-folder $WORKFOLDER \
                      --extra-group-column transition_group_id --extra-group-column transition_group_id

pyprophet-cli subsample --random-seed 43 --data-folder $DATA_FOLDER --data-filename-pattern "$PATTERN" --work-folder $WORKFOLDER --sample-factor 1.0

pyprophet-cli learn --random-seed 43 --work-folder $WORKFOLDER --ignore-invalid-scores

pyprophet-cli apply_weights --work-folder $WORKFOLDER --data-folder $DATA_FOLDER --data-filename-pattern "$PATTERN"

pyprophet-cli score --work-folder $WORKFOLDER --data-folder $DATA_FOLDER --overwrite-results  --d-score-cutoff -100000 --result-folder $RESULTFOLDER --use-fdr  --statistics-mode local --data-filename-pattern "$PATTERN"
