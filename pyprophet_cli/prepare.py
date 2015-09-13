# encoding: utf-8
from __future__ import print_function

import os

import pandas as pd

import io
import core
from common_options import data_folder, separator, data_filename_pattern, work_folder
from constants import SCORE_COLUMNS_FILE, ID_COL
from exceptions import InvalidInput


class Prepare(core.Job):

    """runs validity tests on input files in DATA_FOLDER. this command mostly checks
    if the column names are consistent.
    """

    command_name = "prepare"
    options = [data_folder, separator, data_filename_pattern, work_folder]

    def run(self):
        self._setup_work_folder()
        common_column_names = self._check_headers()
        self._write_score_column_names(common_column_names)

    def _setup_work_folder(self):
        if not os.path.exists(self.work_folder):
            os.makedirs(self.work_folder)

    def _check_headers(self):
        input_file_pathes = io.scan_files(self.data_folder, self.data_filename_pattern)
        headers = set()
        for path in input_file_pathes:
            header = pd.read_csv(path, sep=self.separator, nrows=1).columns
            expected = ID_COL
            if header[0] != expected:
                raise InvalidInput("first column of %s has wrong name %r. exepected %r" %
                                   (path, header[0], expected))
            headers.add(tuple(header))
        if not headers:
            raise InvalidInput("dit not find any data file")
        if len(headers) > 1:
            msg = []
            for header in headers:
                msg.append(", ".join(map(repr(header))))
            raise InvalidInput("found different headers in input_files:" + "\n".join(msg))
        self.logger.info("header check succeeded")
        return header

    def _write_score_column_names(self, names):

        score_columns = [name for name in names if name.startswith("main_") or
                         name.startswith("var_")]
        with open(os.path.join(self.work_folder, SCORE_COLUMNS_FILE), "w") as fp:
            for score_column in score_columns:
                print(score_column, file=fp)
