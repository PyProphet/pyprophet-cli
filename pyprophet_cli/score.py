# encoding: utf-8
from __future__ import print_function

from os.path import join, exists
from os import listdir, makedirs
import shutil

import click

import numpy as np
import pandas as pd


from pyprophet.optimized import rank32
from pyprophet.report import save_report
from pyprophet.stats import (calculate_final_statistics, summary_err_table,
                             lookup_s_and_q_values_from_error_table, final_err_table,
                             lookup_p_values_from_error_table)

from . import io, core

from .common_options import (job_number, job_count, local_folder, separator, data_folder,
                             work_folder, chunk_size, data_filename_pattern, result_folder,
                             lambda_)

from .constants import (SCORE_DATA_FILE_ENDING, TOP_SCORE_DATA_FILE_ENDING, SCORED_ENDING,
                        EXTRA_GROUP_COLUMNS_FILE, ID_COL)

from .exceptions import WorkflowError


def _attach_m_scores(chunk, d_scores, stats, name):
    __, q = lookup_s_and_q_values_from_error_table(d_scores, stats.df)
    nan_count = np.sum(np.isnan(q))
    if name is None:
        chunk["m_score"] = q
    else:
        chunk["%s_m_score" % name] = q
    return nan_count


def _attach_p_values(chunk, d_scores, stats, name):
    p = lookup_p_values_from_error_table(d_scores, stats.df)
    if name is None:
        chunk["p_value"] = p
    else:
        chunk["%s_p_value" % name] = p


def _filter_score_names(chunk):
    score_names = []
    for name in chunk.columns:
        if name.startswith("main_") or name.startswith("var_"):
            score_names.append(name)
        return score_names


class _Scorer(object):

    def _setup(self):
        io.setup_input_files(self, self.data_filename_pattern)
        self._load_extra_score_columns()
        self._load_score_data()
        self._setup_result_folder()
        self._compute_global_stats()

    def _setup_result_folder(self):
        if not self.overwrite_results and exists(self.result_folder):
            if listdir(self.result_folder):
                raise WorkflowError("result folder %r is not empty, you may use "
                                    "--overwrite-results" % self.result_folder)
        if not exists(self.result_folder):
            makedirs(self.result_folder)

    def _load_extra_score_columns(self):
        self.extra_group_columns = io.read_column_names(self.work_folder, EXTRA_GROUP_COLUMNS_FILE)

    def _load_score_data_local(self, i):
        empty = np.empty(0, dtype=np.float32)
        top_scores = {}

        in_path = self._pathes_of_files_for_processing[i]
        path = join(self.work_folder, io.file_name_stem(in_path) + TOP_SCORE_DATA_FILE_ENDING)
        store = pd.HDFStore(path, mode="r")
        for key in store.keys():
            if key not in top_scores:
                top_scores[key] = store[key]
        self.top_scores = top_scores

    def _load_score_data(self):

        empty = np.empty(0, dtype=np.float32)
        top_scores = {}

        for name in listdir(self.work_folder):
            if name.endswith(TOP_SCORE_DATA_FILE_ENDING):
                path = join(self.work_folder, name)
                store = pd.HDFStore(path, mode="r")

                for key in store.keys():
                    if key not in top_scores:
                        top_scores[key] = store[key]
                    else:
                        # compute new data frame by aligning current one to previous one
                        # and computing max score over columns, then cleaning up the column
                        # names so that the next iteration will work again:
                        if self.statistics_mode == "global":
                            # we determine top target transistion over all runs:
                            existing = top_scores[key]
                            new = store[key]
                            m = pd.merge(existing, new, on="ids", how="outer")
                            m["scores"] = m[["scores_x", "scores_y"]].max(axis=1)
                            m["decoy_flags_x"] = m["decoy_flags_x"].fillna(m["decoy_flags_y"])
                            m.drop(["scores_x", "scores_y", "decoy_flags_y"], axis=1, inplace=True)
                            m.rename(columns=dict(decoy_flags_x="decoy_flags"), inplace=True)
                            top_scores[key] = m
                        elif self.statistics_mode == "inbetween":
                            existing = top_scores[key]
                            new = store[key]
                            top_scores[key] = pd.concat((top_scores[key], store[key]))

        if not top_scores:
            raise WorkflowError("no top score data found in %s" % self.work_folder)

        self.top_scores = top_scores

    def _compute_global_stats(self):

        """
        todo: zu den scores noch die ids (taret_id, protein_id, ...) heraus schreiben
        (evtl: hashen, oder mit categories ....)
        für statistik dann die top_targets und top_decoys global bestimmen !
        """

        scores = self.top_scores["/tg_id"]
        top_decoy_scores = scores[scores["decoy_flags"]].scores.values
        top_target_scores = scores[~scores["decoy_flags"]].scores.values

        mean = np.mean(top_decoy_scores)
        std_dev = np.std(top_decoy_scores, ddof=1)
        self.decoy_mean = mean
        self.decoy_std_dev = std_dev

        top_decoy_scores = (top_decoy_scores - mean) / std_dev
        top_target_scores = (top_target_scores - mean) / std_dev

        self.stats = calculate_final_statistics(top_target_scores, top_target_scores,
                                                top_decoy_scores, self.lambda_,
                                                not self.use_fdr)

        if self.job_number == 1:
            err_table = final_err_table(self.stats.df)
            self.cutoffs = err_table["cutoff"].values
            self.svalues = err_table["svalue"].values
            self.qvalues = err_table["qvalue"].values
            self.pvalues = err_table["pvalue"].values
            self.top_target_scores = top_target_scores
            self.top_decoy_scores = top_decoy_scores

        self._log_summary_stats(None, self.stats, top_target_scores, top_decoy_scores)

        self.extra_stats = []
        for name in self.extra_group_columns:
            stats = self._compute_stat_by(name)
            self.extra_stats.append(stats)

    def _compute_stat_by(self, name):

        scores = self.top_scores["/%s" % name]
        top_decoy_scores = scores[scores["decoy_flags"]].scores.values
        top_target_scores = scores[~scores["decoy_flags"]].scores.values

        top_decoy_scores = (top_decoy_scores - self.decoy_mean) / self.decoy_std_dev
        top_target_scores = (top_target_scores - self.decoy_mean) / self.decoy_std_dev

        stats = calculate_final_statistics(top_target_scores, top_target_scores,
                                           top_decoy_scores, self.lambda_,
                                           not self.use_fdr)
        self._log_summary_stats(name, stats, top_target_scores, top_decoy_scores)
        return stats

    def _log_summary_stats(self, group_column, stats, top_target_scores, top_decoy_scores):

        self.logger.info("")
        self.logger.info("STATS WHEN GROUPED BY %s" % (group_column or ID_COL))
        self.logger.info("")
        self.logger.info("num_null   : %.2f" % stats.num_null)
        self.logger.info("num_total  : %.2f" % stats.num_total)
        self.logger.info("stats shape: %s" % (stats.df.shape,))
        self.logger.info("mean top target scores: %3f" % np.mean(top_target_scores))
        self.logger.info("sdev top target scores: %3f" % np.std(top_target_scores, ddof=1))
        self.logger.info("mean top decoy  scores: %3f" % np.mean(top_decoy_scores))
        self.logger.info("sdev top decoy  scores: %3f" % np.std(top_decoy_scores, ddof=1))

        summary_stats = summary_err_table(stats.df)
        self.logger.info("")
        for line in str(summary_stats).split("\n"):
            self.logger.info(line.rstrip())
        self.logger.info("")

        if self.job_number == 1:
            if group_column is None:
                path = join(self.result_folder, "summary_stats.txt")
            else:
                path = join(self.result_folder, "summary_stats_grouped_by_%s.txt" % group_column)
            with open(path, "w") as fp:
                print("num_null   : %.2f" % stats.num_null, file=fp)
                print("num_total  : %.2f" % stats.num_total, file=fp)
                print("stats shape: %s" % (stats.df.shape,), file=fp)
                print("mean top target scores: %3f" % np.mean(top_target_scores), file=fp)
                print("sdev top target scores: %3f" % np.std(top_target_scores, ddof=1), file=fp)
                print("mean top decoy  scores: %3f" % np.mean(top_decoy_scores), file=fp)
                print("sdev top decoy  scores: %3f" % np.std(top_decoy_scores, ddof=1), file=fp)
                print(file=fp)
                summary_stats.to_string(fp)

            try:
                import matplotlib
            except ImportError:
                self.logger.warn("!" * 80)
                self.logger.warn("can not import matplotlib, creating report.pdf is skipped.")
                self.logger.warn("!" * 80)
                return
            try:
                # install prettier plotting styles:
                import seaborn
            except ImportError:
                pass
            if group_column is None:
                path = join(self.result_folder, "report.pdf")
                save_report(path, "", self.top_decoy_scores, self.top_target_scores, self.top_decoy_scores,
                            self.top_target_scores, self.cutoffs, self.svalues, self.qvalues,
                            self.pvalues, self.lambda_)
            else:
                err_table = final_err_table(stats.df)
                path = join(self.result_folder, "report_grouped_by_%s.pdf" % group_column)
                save_report(path, "", top_decoy_scores, top_target_scores, top_decoy_scores,
                            top_target_scores, err_table["cutoff"].values,
                            err_table["svalue"].values, err_table["qvalue"].values,
                            err_table["pvalue"].values, self.lambda_
                            )

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

        self.logger.info("load score data %s" % score_path)

        npzfile = np.load(score_path)
        scores = npzfile["scores"]
        scores = (scores - self.decoy_mean) / self.decoy_std_dev

        numeric_ids = npzfile["numeric_ids"]
        ranks = rank32(numeric_ids, scores)

        row_idx = 0
        score_names = None

        out_path = join(self.result_folder, io.file_name_stem(in_path) + SCORED_ENDING)

        self.logger.info("process %s" % in_path)
        write_header = True

        dtype = io.setup_dtypes(self.work_folder)

        with open(out_path, "w") as fp:

            chunk_count = 0
            for chunk in pd.read_csv(in_path, sep=self.separator, chunksize=self.chunk_size,
                                     dtype=dtype):
                chunk_count += 1
                if score_names is None:
                    score_names = _filter_score_names(chunk)

                chunk.drop(score_names, axis=1, inplace=True)
                nrows = chunk.shape[0]

                d_scores = scores[row_idx: row_idx + nrows]

                chunk["peak_group_rank"] = ranks[row_idx: row_idx + nrows]
                chunk["d_score"] = d_scores

                nan_count = _attach_m_scores(chunk, d_scores, self.stats, None)
                if nan_count:
                    self.logger.warn("found %d NAN in q-scores for transition groups !!" % nan_count)
                _attach_p_values(chunk, d_scores, self.stats, None)

                for (name, stats) in zip(self.extra_group_columns, self.extra_stats):
                    nan_count = _attach_m_scores(chunk, d_scores, stats, name)
                    if nan_count:
                        self.logger.warn("found %d NAN in q-scores for %s !!" % (nan_count, name))
                    _attach_p_values(chunk, d_scores, self.stats, name)

                if self.d_score_cutoff is not None:
                    chunk = chunk[chunk["d_score"] >= self.d_score_cutoff]

                chunk.to_csv(fp, sep=self.separator, header=write_header, index=False)
                row_idx += nrows

                write_header = False

            self.logger.info("processed %d chunks from %s" % (chunk_count, in_path))

        self.logger.info("wrote %s" % out_path)


class _GlobalScorer(_Scorer):

    def _run(self):
        self._setup()
        for i in xrange(self.job_number - 1, len(self.input_file_pathes), self.job_count):
            self._local_job(i)


class Score(core.Job):

    """applies weights to given data files and writes them to WORK_FOLDER
    """

    command_name = "score"
    options = [job_number, job_count, local_folder, separator, data_folder, work_folder,
               chunk_size, data_filename_pattern,
               result_folder,
               click.option("--use-fdr", is_flag=True, help="use FDR, not pFDR for scoring"),
               click.option("--overwrite-results", is_flag=True),
               lambda_,
               click.option("--d-score-cutoff", type=float, default=None,
                            help="filter output files by given d-score threshold"),
               click.option("--statistics-mode", type=click.Choice(['local', 'global', 'inbetween']))]

    def run(self):
        """run processing step from commandline"""
        self.__class__ = _GlobalScorer
        self._run()


