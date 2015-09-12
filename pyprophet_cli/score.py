# encoding: utf-8
from __future__ import print_function

from os.path import join, exists
from os import listdir, makedirs
import shutil

import click

import numpy as np
import pandas as pd


from pyprophet.optimized import find_top_ranked, rank
from pyprophet.stats import (calculate_final_statistics, summary_err_table,
                             lookup_s_and_q_values_from_error_table)

import io

from core import Job
from common_options import (job_number, job_count, local_folder, separator, data_folder,
                            work_folder, chunk_size, data_filename_pattern, result_folder)

from constants import SCORE_DATA_FILE_ENDING
from exceptions import WorkflowError


class Score(Job):

    """applies weights to given data files and writes them to WORK_FOLDER
    """

    command_name = "score"
    options = [job_number, job_count, local_folder, separator, data_folder, work_folder,
               chunk_size, data_filename_pattern,
               result_folder,
               click.option("--overwrite-results", is_flag=True),
               click.option("--lambda", "lambda_", default=0.4,
                      help="lambda value for storeys method [default=0.4]"),
               click.option("--d-score-cutoff", type=float, default=None,
                      help="filter output files by given d-score threshold")]

    def run(self):
        """run processing step from commandline"""
        self._setup()
        for i in xrange(self.job_number - 1, len(self.input_file_pathes), self.job_count):
            self._local_job(i)

    def _setup(self):
        io.setup_input_files(self, self.data_filename_pattern)
        self._load_score_data()
        self._create_global_stats()
        self._setup_result_folder()

    def _setup_result_folder(self):
        if not self.overwrite_results and exists(self.result_folder):
            if listdir(self.result_folder):
                raise WorkflowError("result folder %r is not empty, you may use "
                                    "--overwrite-results" % self.result_folder)

        if not exists(self.result_folder):
            makedirs(self.result_folder)

    def _load_score_data(self):
        all_scores = []
        all_decoy_flags = []
        all_ids = []
        last_max = 0
        for name in listdir(self.work_folder):
            if name.endswith(SCORE_DATA_FILE_ENDING):
                path = join(self.work_folder, name)
                npzfile = np.load(path)

                numeric_ids = npzfile["numeric_ids"]
                numeric_ids += last_max
                last_max = np.max(numeric_ids) + 1
                all_ids.append(numeric_ids)

                scores = npzfile["scores"]
                all_scores.append(scores)

                decoy_flags = npzfile["decoy_flags"]
                all_decoy_flags.append(decoy_flags)

        if not all_scores:
            raise WorkflowError("no score matrices found in %s" % self.work_folder)

        self.scores = np.hstack(all_scores)
        self.numeric_ids = np.hstack(all_ids)
        self.decoy_flags = np.hstack(all_decoy_flags)

    def _create_global_stats(self):

        decoy_scores = self.scores[self.decoy_flags]
        decoy_ids    = self.numeric_ids[self.decoy_flags]

        assert decoy_ids.shape == decoy_scores.shape
        flags = find_top_ranked(decoy_ids, decoy_scores).astype(bool)
        top_decoy_scores = decoy_scores[flags]

        mean = np.mean(top_decoy_scores)
        std_dev = np.std(top_decoy_scores, ddof=1)
        self.decoy_mean = mean
        self.decoy_std = std_dev

        decoy_scores = (decoy_scores - mean) / std_dev

        target_scores = self.scores[~self.decoy_flags]
        target_scores = (target_scores - mean) / std_dev
        target_ids = self.numeric_ids[~self.decoy_flags]

        assert target_ids.shape == target_scores.shape
        flags = find_top_ranked(target_ids, target_scores).astype(bool)
        top_target_scores = target_scores[flags]

        self.stats = calculate_final_statistics(top_target_scores, target_scores, decoy_scores,
                                                self.lambda_)

        summary_stats = summary_err_table(self.stats.df)
        self.logger.info("overall stat")
        for line in str(summary_stats).split("\n"):
            self.logger.info(line.rstrip())

    def _local_job(self, i):
        if self.local_folder:
            self._copy_to_local(i)
        self._score(i)

    def _copy_to_local(self, i):
        path = self.input_file_pathes[i]
        self.logger.info("copy %s to %s" % (path, self.local_folder))
        shutil.copy(path, self.local_folder)

    def _score(self, i):
        in_path = self._pathes_of_files_for_processing[i]

        score_path = join(self.work_folder, io.file_name_stem(in_path) + SCORE_DATA_FILE_ENDING)
        if not exists(score_path):
            raise WorkflowError("file %s does not exist" % score_path)

        # todo
        # fix nps file reading
        # compute and assign peakgroup ranks !
        # complete regression tests for apply_weights and score steps !

        self.logger.info("load scores %s" % score_path)

        npzfile = np.load(score_path)
        scores = npzfile["scores"]
        scores = (scores - self.decoy_mean) / self.decoy_std

        numeric_ids = npzfile["numeric_ids"]
        ranks = rank(numeric_ids, scores)

        row_idx = 0
        score_names = None

        out_path = join(self.result_folder, io.file_name_stem(in_path) + SCORE_DATA_FILE_ENDING)

        self.logger.info("process %s" % in_path)
        write_header = True

        dtype = io.setup_dtypes(self.work_folder)

        with open(out_path, "w") as fp:

            chunk_count = 0
            for chunk in pd.read_csv(in_path, sep=self.separator, chunksize=self.chunk_size,
                                     dtype=dtype):
                chunk_count += 1

                if score_names is None:
                    score_names = []
                    for name in chunk.columns:
                        if name.startswith("main_") or name.startswith("var_"):
                            score_names.append(name)

                chunk.drop(score_names, axis=1, inplace=True)
                nrows = chunk.shape[0]

                # we have  to build a data frame because lookup_s_and_q_values_from_error_table
                # functions expectes this:
                d_scores = scores[row_idx: row_idx + nrows]
                s, q = lookup_s_and_q_values_from_error_table(d_scores, self.stats.df)
                chunk["d_score"] = d_scores
                chunk["m_score"] = q
                chunk["peak_group_rank"] = ranks[row_idx: row_idx + nrows]

                if self.d_score_cutoff is not None:
                    chunk = chunk[chunk["d_score"] >= self.d_score_cutoff]

                chunk.to_csv(fp, sep=self.separator, header=write_header, index=False)
                row_idx += nrows

                write_header = False

            self.logger.info("processed %d chunks from %s" % (chunk_count, in_path))

        self.logger.info("wrote %s" % out_path)

