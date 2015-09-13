# encoding: utf-8
# vim: et sw=4 ts=4
from __future__ import print_function

import os
import random

import pandas as pd
import numpy as np

from pyprophet.pyprophet import PyProphet

from . import io, core
from .common_options import work_folder, separator, random_seed, ignore_invalid_scores
from .constants import SUBSAMPLED_FILES_PATTERN, INVALID_COLUMNS_FILE, WEIGHTS_FILE_NAME

from .exceptions import WorkflowError


join = os.path.join


class Learn(core.Job):

    """runs pyprophet learner on data files in WORK_FOLDER.
    writes weights files in this folder.
    """

    command_name = "learn"
    options = [work_folder, separator, random_seed, ignore_invalid_scores]

    def run(self):
        if self.random_seed:
            self.logger.info("set random seed to %r" % self.random_seed)
            random.seed(self.random_seed)

        pathes = io.scan_files(self.work_folder, SUBSAMPLED_FILES_PATTERN % "*")
        pyprophet = PyProphet()

        self.logger.info("read subsampled files from %s" % self.work_folder)
        tables = list(pyprophet.read_tables_iter(pathes, self.separator))
        self.logger.info("finished reading subsampled files from %s" % self.work_folder)

        # collect colum names for which at least one input table contains only invalid names
        invalid_columns = {}
        for table in tables:
            for name in table.columns:
                col_data = table[name]
                invalid_columns[name] = invalid_columns.get(name) or pd.isnull(col_data).all()

        invalid_column_names = [name for (name, invalid) in invalid_columns.items() if invalid]
        if not self.ignore_invalid_scores and invalid_column_names:
            msg = ", ".join(invalid_column_names)
            raise WorkflowError("columns %s only contain invalid/missing values" % msg)

        with open(os.path.join(self.work_folder, INVALID_COLUMNS_FILE), "w") as fp:
            for name in invalid_column_names:
                print(name, file=fp)

        for table in tables:
            table.drop(invalid_column_names, axis=1, inplace=True)

        self.logger.info("run pyprophet core algorithm")
        result, scorer, weights = pyprophet._learn_and_apply(tables)
        for line in str(result.summary_statistics).split("\n"):
            self.logger.info(line.rstrip())

        self.logger.info("write sum_stat_subsampled.txt")
        with open(join(self.work_folder, "sum_stat_subsampled.txt"), "w") as fp:
            result.summary_statistics.to_csv(fp, sep=self.separator, index=False)

        self.logger.info("write full_stat_subsampled.txt")
        with open(join(self.work_folder, "full_stat_subsampled.txt"), "w") as fp:
            result.final_statistics.to_csv(fp, sep=self.separator, index=False)

        weights_path = join(self.work_folder, WEIGHTS_FILE_NAME)
        self.logger.info("write {}".format(weights_path))
        np.savetxt(weights_path, weights)
