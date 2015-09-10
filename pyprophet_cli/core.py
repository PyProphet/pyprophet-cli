# encoding: utf-8
# vi: et sw=4 ts=4

from __future__ import print_function


import collections
import os
import random
import shutil
import tempfile

import pkg_resources

import click
import numpy as np
import pandas as pd

from . import tools

from pyprophet.pyprophet import PyProphet
from pyprophet.optimized import find_top_ranked, rank
from pyprophet.stats import (calculate_final_statistics, summary_err_table,
                             lookup_s_and_q_values_from_error_table)


join = os.path.join
basename = os.path.basename


def file_name_stem(path):
    return os.path.splitext(basename(path))[0]


Path = click.Path
option = click.option

job_number = option("--job-number", default=1, help="1 <= job-number <= job-count [default=1]")
job_count = option("--job-count", default=1, help="overall number of batch jobs [default=1]")
data_folder = option("--data-folder", help="folder of input data to process",
                     type=Path(exists=True, file_okay=False, dir_okay=True, readable=True),
                     required=True)

work_folder = option("--work-folder",
                        help="folder for intermediate results which are needed by following processing steps",
                        type=Path(file_okay=False, dir_okay=True, readable=True, writable=True),
                        required=True)

result_folder = option("--result-folder",
                        help="folder for final results",
                        type=Path(file_okay=False, dir_okay=True, writable=True),
                        required=True)

local_folder = option("--local-folder",
                      type=Path(file_okay=False, dir_okay=True, readable=True, writable=True),
                      help="local folder on computing node ($TMPDIR for brutus lsf)")

random_seed = option("--random-seed", type=int,
                     help="set a fixed seed for reproducable results")

chunk_size = option("--chunk-size", default=100000,
                    help="chunk size when reading input files, may impact processing speed "
                    "[default=100000]",
                    )

data_filename_pattern = option("--data-filename-pattern", default="*.txt",
                               help="glob pattern to filter files in data folder")

ignore_invalid_scores = option("--ignore-invalid-scores", is_flag=True,
                               help="ignore columns where all values are invalid/missing")


def transform_sep(ctx, param, value):
    return {"tab": "\t", "comma": ",", "semicolon": ";"}.get(value)


separator = option("--separator", type=click.Choice(("tab", "comma", "semicolon")),
                   default="tab", help="separator for input and output files [default=tab]",
                   callback=transform_sep)

sample_factor = option("--sample-factor", required=True, type=float,
                       help="sample factor in range 0.0 .. 1.0")


class InvalidInput(Exception):
    pass


class WorkflowError(Exception):
    pass


WEIGHTS_FILE_NAME = "weights.txt"
SCORE_DATA_FILE_ENDING = "_score_data.npz"
SCORED_ENDING = "_scored.txt"
INVALID_COLUMNS_FILE = "invalid_columns.txt"
SCORE_COLUMNS_FILE = "score_columns.txt"
SUBSAMPLED_FILES_PATTERN = "subsampled_%s.txt"


class JobMeta(type):

    def __new__(cls_, name, parents, dd):
        to_check = "command_name", "options", "run"
        if object not in parents:
            if any(field not in dd for field in to_check):
                raise TypeError("needed attributes/methods: %s" % ", ".join(to_check))
        return super(JobMeta, cls_).__new__(cls_, name, parents, dd)


class Job(object):

    ID_COL = "transition_group_id"

    __metaclass__ = JobMeta


class Prepare(Job):

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
        input_file_pathes = tools.scan_files(self.data_folder, self.data_filename_pattern)
        headers = set()
        for path in input_file_pathes:
            header = pd.read_csv(path, sep=self.separator, nrows=1).columns
            expected = self.ID_COL
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

        score_columns = [name for name in names if name.startswith("main_") or\
                                                   name.startswith("var_")]
        with open(os.path.join(self.work_folder, SCORE_COLUMNS_FILE), "w") as fp:
            for score_column in score_columns:
                print(score_column, file=fp)


def _setup_input_files(job, data_filename_pattern):
    if job.local_folder:
        if not os.path.exists(job.local_folder):
            raise WorkflowError("%s does not exist" % job.local_folder)
    job.input_file_pathes = tools.scan_files(job.data_folder, data_filename_pattern)
    if not job.input_file_pathes:
        raise WorkflowError("data folder %s is empty" % job.data_folder)

    job._pathes_of_files_for_processing = []
    for path in job.input_file_pathes:
        name = basename(path)
        folder = job.local_folder if job.local_folder else job.data_folder
        path = join(folder, name)
        job._pathes_of_files_for_processing.append(path)


def _read_invalid_colums(work_folder):
    with open(os.path.join(work_folder, INVALID_COLUMNS_FILE), "r") as fp:
        for line in fp:
            yield line.rstrip()


def _setup_dtypes(work_folder):
    dtype = {}
    with open(os.path.join(work_folder, SCORE_COLUMNS_FILE), "r") as fp:
        for line in fp:
            dtype[line.rstrip()] = np.float32
    return dtype


class Subsample(Job):

    """subsamples transition groups from given input files in DATA_FOLDER.
    For following procession steps the data is written to the provided WORK_FOLDER.
    """

    command_name = "subsample"
    options = [job_number, job_count, local_folder, separator, data_folder, work_folder,
               random_seed, chunk_size, sample_factor, data_filename_pattern,
               ]

    def run(self):
        """run processing step from commandline"""
        try:
            self.sample_factor = float(self.sample_factor)
        except ValueError:
            raise InvalidInput("sample factor is not a valid float")
        if not 0.0 < self.sample_factor <= 1.0:
            raise InvalidInput("sample factor %r is out of range 0.0 < sample_factor <= 1.0")
        self._setup()
        for i in xrange(self.job_number - 1, len(self.input_file_pathes), self.job_count):
            self._local_job(i)

    def _setup(self):
        _setup_input_files(self, self.data_filename_pattern)

    def _local_job(self, i):
        if self.local_folder:
            self._copy_to_local(i)
        self._subsample(i)

    def _copy_to_local(self, i):
        path = self.input_file_pathes[i]
        self.logger.info("copy %s to %s" % (path, self.local_folder))
        shutil.copy(path, self.local_folder)
        self.logger.info("copied %s to %s" % (path, self.local_folder))

    def _subsample(self, i):
        path = self._pathes_of_files_for_processing[i]

        self.logger.info("start subsample %s" % path)

        ids = []
        overall_line_count = 0
        chunk_count = 0
        usecols = [self.ID_COL, "decoy"]
        for chunk in pd.read_csv(path, sep=self.separator, chunksize=self.chunk_size,
                                 usecols=usecols):
            chunk_count += 1
            ids.extend(chunk[self.ID_COL])
            overall_line_count += len(chunk)
            for name in chunk.columns:
                col_data = chunk[name]

        self.logger.info("read %d chunks from %s" % (chunk_count, path))

        line_counts = collections.Counter(ids)

        ids = set(ids)
        decoys = set(id_[6:] for id_ in ids if id_.startswith("DECOY_"))
        targets = set(id_ for id_ in ids if not id_.startswith("DECOY_"))
        valid_targets = list(decoys & targets)

        # we shuffle the target_ids randomly:
        if self.random_seed:
            self.logger.info("   set random seed to %r" % self.random_seed)
            random.seed(self.random_seed)
        random.shuffle(valid_targets)

        # now we iterate over the ids until the size limit for the result file is achieved:
        consumed_lines = 0
        total_lines_output = overall_line_count * self.sample_factor
        sample_targets = []
        for (i, id_) in enumerate(valid_targets):
            if consumed_lines > total_lines_output:
                break
            consumed_lines += line_counts[id_]
            consumed_lines += line_counts["DECOY_" + id_]
            sample_targets.append(id_)

        # now we create a set of all ids (targets + decoys) we want to keep:
        sample_targets = set(sample_targets)
        sample_decoys = set("DECOY_" + id_ for id_ in sample_targets)
        sample_ids = sample_targets | sample_decoys

        self.logger.info("   now subsample %d target groups from %s" % (len(sample_ids), path))

        # determine name of result file
        stem = file_name_stem(path)
        if self.local_folder:
            out_path = os.path.join(self.local_folder, SUBSAMPLED_FILES_PATTERN % stem)
        else:
            out_path = join(self.work_folder, SUBSAMPLED_FILES_PATTERN % stem)
        self.logger.info("   start to subsample from %s and write to %s" % (path, out_path))

        dtype = _setup_dtypes(self.work_folder)
        # write result file
        with open(out_path, "w") as fp:

            write_header = True
            chunk_count = 0
            for chunk in pd.read_csv(path, sep=self.separator, chunksize=self.chunk_size,
                                     dtype=dtype):
                chunk_count += 1
                chunk = chunk[chunk[self.ID_COL].isin(sample_ids)]
                chunk.to_csv(fp, sep=self.separator, header=write_header, index=False)
                write_header = False

        self.logger.info("   read %d chunks from %s" % (chunk_count, path))
        self.logger.info("   wrote %s" % out_path)

        if self.local_folder:
            self.logger.info("   copy subsampled data from local folder to work folder")
            shutil.copy(out_path, self.work_folder)
            self.logger.info("   copied subsampled data from local folder to work folder")

        self.logger.info("subsampling %s finished" % path)



class Learn(Job):

    """runs pyprophet learner on data files in WORK_FOLDER.
    writes weights files in this folder.
    """

    command_name = "learn"
    options = [work_folder, separator, random_seed, ignore_invalid_scores]

    def run(self):
        if self.random_seed:
            self.logger.info("set random seed to %r" % self.random_seed)
            random.seed(self.random_seed)


        pathes = tools.scan_files(self.work_folder, SUBSAMPLED_FILES_PATTERN % "*")
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
        _setup_input_files(self, self.data_filename_pattern)
        self._load_weights()

    def _load_weights(self):
        weights_path = join(self.work_folder, WEIGHTS_FILE_NAME)
        if not os.path.exists(weights_path):
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
        invalid_columns = list(_read_invalid_colums(self.work_folder))
        chunk_count = 0

        dtype = _setup_dtypes(self.work_folder)

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

        numeric_ids = map(tg_numeric_ids.get, tg_ids)
        decoy_flags = map(lambda tg_id: tg_id.startswith("DECOY_"), tg_ids)

        stem = file_name_stem(path)
        out_path = join(self.work_folder, stem + SCORE_DATA_FILE_ENDING)
        np.savez(out_path, numeric_ids=numeric_ids, decoy_flags=decoy_flags,
                           scores=np.hstack(all_scores))
        self.logger.info("wrote %s" % out_path)


class Score(Job):

    """applies weights to given data files and writes them to WORK_FOLDER
    """

    command_name = "score"
    options = [job_number, job_count, local_folder, separator, data_folder, work_folder,
               chunk_size, data_filename_pattern,
               result_folder, 
               option("--overwrite-results", is_flag=True),
               option("--lambda", "lambda_", default=0.4,
                      help="lambda value for storeys method [default=0.4]"),
               option("--d-score-cutoff", type=float, default=None,
                      help="filter output files by given d-score threshold")]

    def run(self):
        """run processing step from commandline"""
        self._setup()
        for i in xrange(self.job_number - 1, len(self.input_file_pathes), self.job_count):
            self._local_job(i)

    def _setup(self):
        _setup_input_files(self, self.data_filename_pattern)
        self._load_score_data()
        self._create_global_stats()

    def _load_score_data(self):
        all_scores = []
        all_decoy_flags = []
        all_ids = []
        last_max = 0
        for name in os.listdir(self.work_folder):
            if name.endswith(SCORE_DATA_FILE_ENDING):
                path = join(self.work_folder, name)
                npzfile = np.load(path)

                numeric_ids = npzfile["numeric_ids"]
                numeric_ids += last_max
                last_max = np.max(numeric_ids)
                all_ids.append(numeric_ids)

                scores = npzfile["scores"]
                all_scores.append(scores)

                decoy_flags = npzfile["decoy_flags"]
                all_decoy_flags.append(decoy_flags)

        if not all_scores:
            raise WorkflowError("no score matrices found in %s" % self.work_folder)

        self.scores = np.hstack(all_scores)
        self.ids = np.hstack(all_ids)
        self.decoy_flags = np.hstack(all_decoy_flags)

    def _create_global_stats(self):

        # we precautiously sort the ids and apply the same permutation to
        # the decoy flags and scores:
        assert np.all(sorted(self.ids) == self.ids), "incoming transition groups were scattered in file"

        # perm = np.argsort(self.ids)
        # self.ids = self.ids[perm]
        # self.decoy_flags = self.decoy_flags[perm]
        # self.scores = self.scores[perm]

        decoy_scores = self.scores[self.decoy_flags]
        decoy_ids    = self.ids[self.decoy_flags]

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
        target_ids = self.ids[~self.decoy_flags]

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

        score_path = join(self.work_folder, file_name_stem(in_path) + SCORE_DATA_FILE_ENDING)
        if not os.path.exists(score_path):
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

        if not self.overwrite_results and os.path.exists(self.result_folder):
            if os.listdir(self.result_folder):
                raise WorkflowError("result folder is not empty, you may use --overwrite-results")

        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

        out_path = join(self.result_folder, file_name_stem(in_path) + SCORED_ENDING)

        self.logger.info("process %s" % in_path)
        write_header = True

        dtype = _setup_dtypes(self.work_folder)

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
                # chunk_scores = pd.DataFrame(dict(scores=d_scores))
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


def _load_drivers():

    for ep in pkg_resources.iter_entry_points("pyprophet_cli_plugin", name="config"):
        try:
            driver = ep.load()
        except Exception:
            raise
            raise WorkflowError("driver %s can not be loaded" % ep)
        try:
            name, options, run, help_ = driver()
        except Exception:
            raise
            raise WorkflowError("driver %s can not be loaded" % ep)
        yield _create_run_job(name, options, run, help_)


def _create_run_job(name, options, job_function, help_):

    class _RunWorkflow(Job):
        options = command_name = None

        def run(self):
            job_function(self)

    _RunWorkflow.options = options
    _RunWorkflow.command_name = name
    _RunWorkflow.__doc__ = help_
    _RunWorkflow.run = job_function

    return _RunWorkflow

for driver in _load_drivers():
    pass
