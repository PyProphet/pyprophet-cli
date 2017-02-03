# encoding: utf-8
from __future__ import print_function

from os.path import join, exists, basename
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
                             lambda_, statistics_mode)

from .constants import (SCORE_DATA_FILE_ENDING, TOP_SCORE_DATA_FILE_ENDING, SCORED_ENDING,
                        EXTRA_GROUP_COLUMNS_FILE, ID_COL)

from .exceptions import WorkflowError, CommandLineError


def _filter_score_names(chunk):
    score_names = []
    for name in chunk.columns:
        if name.startswith("main_") or name.startswith("var_"):
            score_names.append(name)
        return score_names


class _Scorer(object):

    def run(self):
        self._setup()
        for run_idx in xrange(self.job_number - 1, len(self.input_file_pathes), self.job_count):
            self._local_job(run_idx)

    def _setup(self):
        self._scan_input_files(self.data_filename_pattern)
        self._load_extra_score_columns()
        self._setup_result_folder()
        self._special_setup()

    def _local_job(self, run_idx):
        if self.local_folder:
            self._copy_to_local(run_idx)
        self._prepare_score(run_idx)
        self._score_run(run_idx)

    def _scan_input_files(self, data_filename_pattern):
        if self.local_folder:
            if not exists(self.local_folder):
                raise WorkflowError("%s does not exist" % self.local_folder)
        self.input_file_pathes = io.scan_files(
            self.data_folder, data_filename_pattern)
        if not self.input_file_pathes:
            raise WorkflowError("data folder %s is empty" % self.data_folder)

    def _load_extra_score_columns(self):
        extra_group_columns = io.read_column_names(
            self.work_folder, EXTRA_GROUP_COLUMNS_FILE)
        self.group_columns = [ID_COL] + extra_group_columns

    def _setup_result_folder(self):
        if not self.overwrite_results and exists(self.result_folder):
            if listdir(self.result_folder):
                raise WorkflowError("result folder %r is not empty, you may use "
                                    "--overwrite-results" % self.result_folder)
        if not exists(self.result_folder):
            makedirs(self.result_folder)

    def _copy_to_local(self, run_idx):
        path = self.input_file_pathes[run_idx]
        self.logger.info("copy %s to %s" % (path, self.local_folder))
        shutil.copy(path, self.local_folder)
        self.input_file_pathes[run_idx] = join(
            self.local_folder, basename(path))

    def _score_run(self, run_idx):

        in_path = self.input_file_pathes[run_idx]
        out_path = join(
            self.result_folder, io.file_name_stem(in_path) + SCORED_ENDING)

        self.logger.info("process %s" % in_path)

        d_scores, ranks = self._load_scores_of(run_idx)
        dtype = io.setup_dtypes(self.work_folder)
        score_names = None
        write_header = True
        row_idx = 0
        with open(out_path, "w") as fp:
            chunk_count = 0
            for chunk in pd.read_csv(in_path, sep=self.separator, chunksize=self.chunk_size,
                                     dtype=dtype):
                chunk_count += 1
                if score_names is None:
                    score_names = _filter_score_names(chunk)
                # remove score columns
                chunk.drop(score_names, axis=1, inplace=True)
                # add new score columns
                self._score_chunk(chunk, d_scores, ranks, row_idx)

                chunk.to_csv(
                    fp, sep=self.separator, header=write_header, index=False)
                write_header = False
                row_idx += chunk.shape[0]
            self.logger.info("processed %d chunks from %s" %
                             (chunk_count, in_path))
        self.logger.info("wrote %s" % out_path)

    def _load_scores_of(self, run_idx):

        in_path = self.input_file_pathes[run_idx]
        score_path = join(
            self.work_folder, io.file_name_stem(in_path) + SCORE_DATA_FILE_ENDING)
        if not exists(score_path):
            raise WorkflowError("file %s does not exist" % score_path)

        self.logger.info("load score data %s" % score_path)

        npzfile = np.load(score_path)
        scores = npzfile["scores"]
        d_scores = (scores - self.decoy_mean) / self.decoy_std_dev
        numeric_ids = npzfile["numeric_ids"]
        ranks = rank32(numeric_ids, d_scores)
        return d_scores, ranks

    def _score_chunk(self, chunk, d_scores, ranks, row_idx):

        nrows = chunk.shape[0]
        d_scores = d_scores[row_idx: row_idx + nrows]
        ranks = ranks[row_idx: row_idx + nrows]

        chunk["peak_group_rank"] = ranks
        chunk["d_score"] = d_scores

        for (group_column_name, stats) in self.stats.items():
            # add column with q values (m-score):
            __, q = lookup_s_and_q_values_from_error_table(d_scores,
                                                           self.stats[group_column_name].df)
            chunk["%s_m_score" % group_column_name] = q
            nan_count = np.sum(np.isnan(q))
            if nan_count:
                self.logger.warn("found %d NAN in q-scores for %s !!" % (nan_count,
                                                                         group_column_name))
            # add column with p values
            p = lookup_p_values_from_error_table(
                d_scores, self.stats[group_column_name].df)
            chunk["%s_p_value" % group_column_name] = p

        if self.d_score_cutoff is not None:
            chunk = chunk[chunk["d_score"] >= self.d_score_cutoff]

    def _compute_stats(self, run_idx=None):
        """None means: global stats, not per run"""

        scores = self.top_scores["/" + ID_COL]
        top_decoy_scores = scores[scores["decoy_flags"]].scores.values

        mean = np.mean(top_decoy_scores)
        std_dev = np.std(top_decoy_scores, ddof=1)
        self.decoy_mean = mean
        self.decoy_std_dev = std_dev

        self.stats = {}
        for group_column_name in self.group_columns:
            self.stats[group_column_name] = self._compute_stat_by(
                run_idx, group_column_name)

    def _compute_stat_by(self, run_idx, name):
        """None means: global stats, not per run"""

        scores = self.top_scores["/%s" % name]
        top_decoy_scores = scores[scores["decoy_flags"]].scores.values
        top_target_scores = scores[~scores["decoy_flags"]].scores.values

        top_decoy_scores = (
            top_decoy_scores - self.decoy_mean) / self.decoy_std_dev
        top_target_scores = (
            top_target_scores - self.decoy_mean) / self.decoy_std_dev

        stats, pvalues = calculate_final_statistics(top_target_scores, top_target_scores,
                                                    top_decoy_scores, self.lambda_,
                                                    self.use_pemp,
                                                    not self.use_fdr)
        self._report_results(
            run_idx, name, stats, top_target_scores, top_decoy_scores, pvalues)
        return stats

    def _report_results(self, run_idx, group_column, stats, top_target_scores, top_decoy_scores,
                        pvalues):
        """None means: global stats, not per run"""

        self._log_header(
            group_column, stats, top_target_scores, top_decoy_scores)
        summary_stats = summary_err_table(stats.df)
        self._log_summary_stats(summary_stats)

        # need only be written once:
        if self.job_number == 1 and run_idx in(0, None):
            self._write_reports(
                stats, summary_stats, group_column, top_target_scores, top_decoy_scores, pvalues)

    def _log_header(self, group_column, stats, top_target_scores, top_decoy_scores):
        self.logger.info("")
        self.logger.info("STATS WHEN GROUPED BY %s" % group_column)
        self.logger.info("")
        self.logger.info("num_null   : %.2f" % stats.num_null)
        self.logger.info("num_total  : %.2f" % stats.num_total)
        self.logger.info("stats shape: %s" % (stats.df.shape,))
        self.logger.info("mean top target scores: %3f" %
                         np.mean(top_target_scores))
        self.logger.info("sdev top target scores: %3f" %
                         np.std(top_target_scores, ddof=1))
        self.logger.info("mean top decoy  scores: %3f" %
                         np.mean(top_decoy_scores))
        self.logger.info("sdev top decoy  scores: %3f" %
                         np.std(top_decoy_scores, ddof=1))

    def _log_summary_stats(self, summary_stats):
        self.logger.info("")
        for line in str(summary_stats).split("\n"):
            self.logger.info(line.rstrip())
        self.logger.info("")
        return summary_stats

    def _write_reports(self, stats, summary_stats, group_column, top_target_scores,
            top_decoy_scores, pvalues, title=None):

        self._write_summary_stats(
            stats, summary_stats, group_column, top_target_scores, top_decoy_scores, title)
        self._write_pdf_report(
            stats, summary_stats, group_column, top_target_scores, top_decoy_scores, pvalues, title)

    def _write_summary_stats(self, stats, summary_stats, group_column, top_target_scores, top_decoy_scores, title):
        if title is None:
            infix = "_"
        else:
            infix = "_%s_" % title

        path = join(
            self.result_folder, "summary_stats%sgrouped_by_%s.txt" % (infix, group_column))

        with open(path, "w") as fp:
            print("num_null   : %.2f" % stats.num_null, file=fp)
            print("num_total  : %.2f" % stats.num_total, file=fp)
            print("stats shape: %s" % (stats.df.shape,), file=fp)
            print("mean top target scores: %3f" %
                  np.mean(top_target_scores), file=fp)
            print("sdev top target scores: %3f" %
                  np.std(top_target_scores, ddof=1), file=fp)
            print("mean top decoy  scores: %3f" %
                  np.mean(top_decoy_scores), file=fp)
            print("sdev top decoy  scores: %3f" %
                  np.std(top_decoy_scores, ddof=1), file=fp)
            print(file=fp)
            summary_stats.to_string(fp)

    def _write_pdf_report(self, stats, summary_stats, group_column, top_target_scores,
                          top_decoy_scores, pvalues, title):
        if title is None:
            postfix = ""
        else:
            postfix = "_%s" % title

        name = "report%s_grouped_by_%s.pdf" % (postfix, group_column)
        path = join(self.result_folder, name)

        err_table = final_err_table(stats.df)
        cutoffs = err_table["cutoff"].values
        svalues = err_table["svalue"].values
        qvalues = err_table["qvalue"].values

        self._setup_plotting_packages()
        save_report(path, "", top_decoy_scores, top_target_scores, top_decoy_scores,
                    top_target_scores, cutoffs, svalues, qvalues,
                    pvalues, self.lambda_)

    def _setup_plotting_packages(self):
        try:
            import matplotlib   # noqa
        except ImportError:
            self.logger.warn("!" * 80)
            self.logger.warn(
                "can not import matplotlib, creating report.pdf is skipped.")
            self.logger.warn("!" * 80)
            return
        try:
            # install prettier plotting styles if available:
            import seaborn     # noqa
        except ImportError:
            pass


class _GlobalScorer(_Scorer):

    def _special_setup(self):
        self._load_and_merge_all_scores()
        self._compute_stats()

    def _prepare_score(self, run_idx):
        pass

    def _load_and_merge_all_scores(self):
        top_scores = {}
        for name in listdir(self.work_folder):
            if name.endswith(TOP_SCORE_DATA_FILE_ENDING):
                path = join(self.work_folder, name)
                with pd.HDFStore(path, mode="r") as store:

                    for key in store.keys():
                        if key not in top_scores:
                            top_scores[key] = store[key]
                        else:
                            top_scores[key] = self._merge_score_frames(
                                top_scores[key], store[key])

        if not top_scores:
            raise WorkflowError(
                "no top score data found in %s" % self.work_folder)

        self.top_scores = top_scores


class _GlobalGlobalScorer(_GlobalScorer):

    def _merge_score_frames(self, existing, new):
        m = pd.merge(existing, new, on="ids", how="outer")
        m["scores"] = m[["scores_x", "scores_y"]].max(axis=1)
        m["decoy_flags_x"] = m["decoy_flags_x"].fillna(m["decoy_flags_y"])
        m.drop(["scores_x", "scores_y", "decoy_flags_y"], axis=1, inplace=True)
        m.rename(columns=dict(decoy_flags_x="decoy_flags"), inplace=True)
        return m


class _LocalGlobalScorer(_GlobalScorer):

    def _merge_score_frames(self, existing, new):
        return pd.concat((existing, new))


class _LocalScorer(_Scorer):

    def _special_setup(self):
        pass

    def _prepare_score(self, run_idx):
        self._load_top_scores(run_idx)
        self._compute_stats(run_idx)

    def _load_top_scores(self, run_idx):
        top_scores = {}

        in_path = self.input_file_pathes[run_idx]
        path = join(self.work_folder, io.file_name_stem(
            in_path) + TOP_SCORE_DATA_FILE_ENDING)
        store = pd.HDFStore(path, mode="r")
        for key in store.keys():
            if key not in top_scores:
                top_scores[key] = store[key]
        self.top_scores = top_scores

    def _report_results(self, run_idx, group_column, stats, top_target_scores, top_decoy_scores,
                        pvalues):

        stem = io.file_name_stem(self.input_file_pathes[run_idx])
        self.logger.info("STATS FOR SCORING %s" % stem)

        self._log_header(
            group_column, stats, top_target_scores, top_decoy_scores)
        self._log_summary_stats(stats)
        summary_stats = summary_err_table(stats.df)
        self._write_reports(
            stats, summary_stats, group_column, top_target_scores, top_decoy_scores, pvalues, stem)


class Score(core.Job):

    """applies weights to given data files and writes them to WORK_FOLDER
    """

    command_name = "score"
    options = [job_number, job_count, local_folder, separator, data_folder, work_folder,
               chunk_size, data_filename_pattern,
               result_folder,
               click.option(
                   "--use-fdr", is_flag=True, help="use FDR, not pFDR for scoring"),
               click.option(
                   "--use-pemp", is_flag=True, help="use empirical p-values instead of p-values "
                                                    "from normal distribution"),
               click.option("--overwrite-results", is_flag=True),
               lambda_,
               click.option("--d-score-cutoff", type=float, default=None,
                            help="filter output files by given d-score threshold"),
               statistics_mode,
               ]

    def run(self):
        """run processing step from commandline"""

        if self.statistics_mode is None:
            raise CommandLineError("--statistics-mode option must be set !")

        # we switch the class, this means: all attributes are kept, but the methods after
        # this assignment will change:
        self.__class__ = {"run-specific": _LocalScorer,
                          "global": _GlobalGlobalScorer,
                          "experiment-wide": _LocalGlobalScorer}[self.statistics_mode]
        self.run()
