# encoding: utf-8
from __future__ import print_function

from os.path import join, exists
import shutil

import numpy as np
import pandas as pd

from .common_options import (job_number, job_count, local_folder, separator, data_folder,
                             work_folder, chunk_size, data_filename_pattern)

from . import io, core

from .constants import (WEIGHTS_FILE_NAME, SCORE_DATA_FILE_ENDING, ID_COL, INVALID_COLUMNS_FILE,
                       EXTRA_GROUP_COLUMNS_FILE, TOP_SCORE_DATA_FILE_ENDING)

from .exceptions import WorkflowError


class ApplyWeights(core.Job):

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
        invalid_columns = io.read_column_names(self.work_folder, INVALID_COLUMNS_FILE)
        chunk_count = 0

        dtype = io.setup_dtypes(self.work_folder)

        extra_group_columns = io.read_column_names(self.work_folder, EXTRA_GROUP_COLUMNS_FILE)

        all_extra_ids = [[] for __ in extra_group_columns]

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

            tg_ids.extend(chunk[ID_COL])
            for i, extra_group_column in enumerate(extra_group_columns):
                all_extra_ids[i].extend(chunk[extra_group_column])

            chunk = chunk.iloc[:, score_column_indices]
            scores = chunk.dot(self.weights).astype(np.float32)
            all_scores.append(scores)

        self.logger.info("read %d chunks from %s" % (chunk_count, path))

        def _assign_numeric_ids(str_ids):
            id_map = {}
            last_id = 0
            for str_id in str_ids:
                if str_id not in id_map:
                    id_map[str_id] = last_id
                    last_id += 1
            return np.array(map(id_map.get, str_ids), dtype=np.int32)

        numeric_ids = _assign_numeric_ids(tg_ids)
        assert np.all(numeric_ids == sorted(numeric_ids)),\
            "incoming transition group ids are scattered over file !"

        # setup dict for assigning extra_group_column integer ids:
        extra_grouping_ids = []
        for extra_ids in all_extra_ids:
            extra_grouping_ids.append(_assign_numeric_ids(extra_ids))

        # extra_grouping_ids = np.array(extra_grouping_ids)
        decoy_flags = map(lambda tg_id: tg_id.startswith("DECOY_"), tg_ids)

        scores = np.hstack(all_scores)

        def extract_top_scores(df):
            decoys = df[df["decoy_flags"]]
            targets = df[~df["decoy_flags"]]
            tt_scores = targets.groupby("ids")["scores"].max().values
            td_scores = decoys.groupby("ids")["scores"].max().values
            return tt_scores, td_scores

        top_scores = {}

        df = pd.DataFrame(dict(scores=scores, ids=numeric_ids, decoy_flags=decoy_flags))
        tt_scores, td_scores = extract_top_scores(df)
        top_scores["top_target_scores"] = tt_scores
        top_scores["top_decoy_scores"] = td_scores

        for name, ids in zip(extra_group_columns, extra_grouping_ids):
            df["ids"] = ids
            tt_scores, td_scores = extract_top_scores(df)
            top_scores["top_target_scores_%s" % name] = tt_scores
            top_scores["top_decoy_scores_%s" % name] = td_scores

        stem = io.file_name_stem(path)
        out_path = join(self.work_folder, stem + TOP_SCORE_DATA_FILE_ENDING)
        np.savez(out_path, **top_scores)
        self.logger.info("wrote %s" % out_path)

        out_path = join(self.work_folder, stem + SCORE_DATA_FILE_ENDING)
        np.savez(out_path, scores=scores, numeric_ids=numeric_ids)
        self.logger.info("wrote %s" % out_path)
