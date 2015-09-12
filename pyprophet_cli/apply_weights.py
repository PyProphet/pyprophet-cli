# encoding: utf-8
from __future__ import print_function

from os.path import join, exists
import shutil

import numpy as np
import pandas as pd

from core import Job
from common_options import (job_number, job_count, local_folder, separator, data_folder,
                            work_folder, chunk_size, data_filename_pattern)
import io
from constants import WEIGHTS_FILE_NAME, SCORE_DATA_FILE_ENDING
from exceptions import WorkflowError


class ApplyWeights(Job):

    """applies weights to given data files and writes them to WORK_FOLDER
    """

    command_name = "apply_weights"
    options = [job_number, job_count, local_folder, separator, data_folder, work_folder,
               chunk_size, data_filename_pattern]

    def run(self):
        """run processing step from commandline"""
        self._setup()
        for i in xrange(self.job_number - 1, len(self.input_file_pathes), self.job_count):
            self._local_job(i)

    def _setup(self):
        io.setup_input_files(self, self.data_filename_pattern)
        self._load_weights()

    def _load_weights(self):
        weights_path = join(self.work_folder, WEIGHTS_FILE_NAME)
        if not exists(weights_path):
            raise WorkflowError("did not find file %s, maybe one of the previous steps of the "
                                "pyprophet workflow is broken" % weights_path)
        try:
            self.weights = np.loadtxt(weights_path)
        except ValueError:
            raise WorkflowError("weights file %s is not valid" % weights_path)

    def _local_job(self, i):
        if self.local_folder:
            self._copy_to_local(i)
        self._compute_scores(i)

    def _copy_to_local(self, i):
        path = self.input_file_pathes[i]
        self.logger.info("copy %s to %s" % (path, self.local_folder))
        shutil.copy(path, self.local_folder)

    def _compute_scores(self, i):
        path = self._pathes_of_files_for_processing[i]

        self.logger.info("start scoring %s" % path)

        all_scores = []
        score_column_indices = None
        tg_ids = []
        invalid_columns = io.read_invalid_colums(self.work_folder)
        chunk_count = 0

        dtype = io.setup_dtypes(self.work_folder)

        for chunk in pd.read_csv(path, sep=self.separator, chunksize=self.chunk_size,
                                 dtype=dtype):
            chunk_count += 1
            if score_column_indices is None:
                score_column_indices = []
                for (i, name) in enumerate(chunk.columns):
                    if name in invalid_columns:
                        continue
                    if name.startswith("main_") or name.startswith("var_"):
                        score_column_indices.append(i)

            tg_ids.extend(chunk[self.ID_COL])
            chunk = chunk.iloc[:, score_column_indices]
            scores = chunk.dot(self.weights).astype(np.float32)
            all_scores.append(scores)

        self.logger.info("read %d chunks from %s" % (chunk_count, path))

        # setup dict assigning target_group_ids to increasing integer numbers:
        tg_numeric_ids = {}
        last_id = 0
        for tg_id in tg_ids:
            if tg_id not in tg_numeric_ids:
                tg_numeric_ids[tg_id] = last_id
                last_id += 1

        numeric_ids = np.array(map(tg_numeric_ids.get, tg_ids))
        assert np.all(numeric_ids == sorted(numeric_ids)),\
            "incoming transition group ids are scattered over file !"

        decoy_flags = map(lambda tg_id: tg_id.startswith("DECOY_"), tg_ids)

        stem = io.file_name_stem(path)
        out_path = join(self.work_folder, stem + SCORE_DATA_FILE_ENDING)
        np.savez(out_path, numeric_ids=numeric_ids, decoy_flags=decoy_flags,
                 scores=np.hstack(all_scores))
        self.logger.info("wrote %s" % out_path)
