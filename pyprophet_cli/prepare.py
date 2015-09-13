# encoding: utf-8
from __future__ import print_function

import os

import pandas as pd

from . import io
from . import core
from .common_options import (data_folder, separator, data_filename_pattern, work_folder,
                             extra_group_columns)
from .constants import SCORE_COLUMNS_FILE, ID_COL, EXTRA_GROUP_COLUMNS_FILE
from .exceptions import InvalidInput


class Prepare(core.Job):

    """runs validity tests on input files in DATA_FOLDER. this command mostly checks
    if the column names are consistent.
    """

    command_name = "prepare"
    options = [data_folder, separator, data_filename_pattern, work_folder, extra_group_columns]

    def run(self):
        self._setup_work_folder()
        self._check_headers()
        self._write_score_column_names()
        self._write_extra_group_column_names()

    def _setup_work_folder(self):
        if not os.path.exists(self.work_folder):
            os.makedirs(self.work_folder)

    def _check_headers(self):
        input_file_pathes = io.scan_files(self.data_folder, self.data_filename_pattern)
        headers = set()
        expected = (ID_COL, ) + self.extra_group_columns
        for path in input_file_pathes:
            header = pd.read_csv(path, sep=self.separator, nrows=1).columns
            missing = [e for e in expected if e not in header]
            if missing:
                raise InvalidInput("header of %r not valid: columns %s are missing" %
                                   (path, ", ".join(missing)))
            headers.add(tuple(header))
        if not headers:
            raise InvalidInput("dit not find any data file")
        if len(headers) > 1:
            msg = []
            for header in headers:
                msg.append(", ".join(map(repr(header))))
            raise InvalidInput("found different headers in input_files:" + "\n".join(msg))
        self.logger.info("header check succeeded")
        self.header = header

    def _write_score_column_names(self):
        score_columns = [name for name in self.header if name.startswith("main_") or
                         name.startswith("var_")]
        io.write_column_names(score_columns, self.work_folder, SCORE_COLUMNS_FILE)

    def _write_extra_group_column_names(self):
        io.write_column_names(self.extra_group_columns, self.work_folder, EXTRA_GROUP_COLUMNS_FILE)
