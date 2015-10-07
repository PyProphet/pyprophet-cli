# encoding: utf-8
from __future__ import print_function

SCORE_DATA_FILE_ENDING = "_score_data.npz"
TOP_SCORE_DATA_FILE_ENDING = "_top_score_data.npz"
SCORED_ENDING = "_scored.txt"

WEIGHTS_FILE_NAME = "weights.txt"
INVALID_COLUMNS_FILE = "invalid_columns.txt"
SCORE_COLUMNS_FILE = "score_columns.txt"
EXTRA_GROUP_COLUMNS_FILE = "extra_group_columns.txt"

SUBSAMPLED_FILES_PATTERN = "subsampled_%s.txt"

# you better do not touch this: the imported pyprophet in learn.py asserts this name:
ID_COL = "transition_group_id"
