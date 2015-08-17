# vi: et sw=4 ts=4

from __future__ import print_function


import collections
import os
import random
import shutil
import tempfile

import click
import pandas

from . import tools

from pyprophet.pyprophet import PyProphet

Path = click.Path
option = click.option

job_number = option("--job-number", type=int, required=True)
job_count = option("--job-count", type=int, required=True)
data_folder = option("--data-folder",
                     type=Path(exists=True, file_okay=False, dir_okay=True, readable=True),
                     required=True)

working_folder = option("--working-folder",
                        type=Path(file_okay=False, dir_okay=True, readable=True, writable=True),
                        required=True)

run_local = option("--run-local", type=bool, default=False)

random_seed = option("--random-seed", type=int)



class InvalidInput(Exception):
    pass


"""
TODO:
    - sep as commandline flag !?
"""

class JobMeta(type):


    def __new__(cls_, name, parents, dd):
        to_check = "requires", "command_name", "options", "run"
        # print(cls_, name, parents, dd.keys())
        if object not in parents:
            if any(field not in dd for field in to_check):
                raise TypeError("needed attributes/methods: %s" % ", ".join(to_check))
        return super(JobMeta, cls_).__new__(cls_, name, parents, dd)


class Job(object):

    SEP = "\t"
    CHUNK_SIZE = 10000
    ID_COL = "transition_group_id"

    __metaclass__ = JobMeta


class CheckInputs(Job):

    """runs validity tests on input files in DATA_FOLDER. this command mostly checks
    if the column names are consistent.
    """

    requires = None
    command_name = "check"
    options = [data_folder]

    def run(self):
        self._check_headers()

    def _check_headers(self):
        input_file_pathes = tools.scan_files(self.data_folder)
        headers = set()
        for path in input_file_pathes:
            header = pandas.read_csv(path, sep=self.SEP, nrows=1).columns
            expected = self.ID_COL
            if header[0] != expected:
                raise InvalidInput("first column of %s has wrong name %r. exepected %r" %
                                   (path, header[0], expected))
            headers.add(tuple(header))
        if len(headers) > 1:
            msg = []
            for header in headers:
                msg.append(", ".join(map(repr(header))))
            raise InvalidInput("found different headers in input_files:" + "\n".join(msg))
        self.logger.info("header check succeeded")


class Subsample(Job):

    """subsamples transition groups from given input files in DATA_FOLDER.
    For following procession steps the data is written to the provided WORKING_FOLDER.
    """

    requires = None
    command_name = "subsample"
    options = [job_number, job_count, data_folder, working_folder, run_local,
               option("--sample-factor", required=True, type=float),
               random_seed,
               ]

    def run(self):
        """run processing step from commandline"""
        self._setup()
        for i in xrange(self.job_number - 1, len(self.input_file_pathes), self.job_count):
            self._local_job(i)

    def _setup(self):
        if self.run_local:
            self.tmp_dir = tempfile.mkdtemp()
        else:
            self.tmp_dir = os.environ.get("TMPDIR")
            assert self.tmp_dir is not None, "$TMPDIR not set !!?!"
        if not os.path.exists(self.working_folder):
            os.makedirs(self.working_folder)
        self.input_file_pathes = tools.scan_files(self.data_folder)
        if not self.input_file_pathes:
            raise InvalidInput("data folder %s is empty" % self.data_folder)


    def _local_job(self, i):
        self._local_file_pathes = []
        self._copy_to_local(i)
        self._subsample(i)

    def _copy_to_local(self, i):
        path = self.input_file_pathes[i]
        self.logger.info("copy %s to %s" % (path, self.tmp_dir))
        shutil.copy(path, self.tmp_dir)
        local_path = os.path.join(self.tmp_dir, os.path.basename(path))
        self._local_file_pathes.append(local_path)

    def _subsample(self, i):
        path = self._local_file_pathes[i]

        self.logger.info("start subsample %s" % path)

        ids = []
        overall_line_count = 0
        for chunk in pandas.read_csv(path, sep=self.SEP, chunksize=self.CHUNK_SIZE):
            ids.extend(chunk[self.ID_COL])
            overall_line_count += len(chunk)

        line_counts = collections.Counter(ids)

        ids = set(ids)
        decoys = set(id_[6:] for id_ in ids if id_.startswith("DECOY_"))
        targets = set(id_ for id_ in ids if not id_.startswith("DECOY_"))
        valid_targets = list(decoys & targets)

        # we shuffle the target_ids randomly:
        if self.random_seed:
            self.logger.info("set random seed to %r" % self.random_seed)
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

        self.logger.info("subsample %d target groups from %s" % (len(sample_ids), path))

        # determine name of result file
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(self.working_folder, "subsampled_%s.txt" % stem)
        self.logger.info("write to %s" % out_path)

        # write result file
        with open(out_path, "w") as fp:

            write_header = True
            for chunk in pandas.read_csv(path, sep=self.SEP, chunksize=self.CHUNK_SIZE):
                chunk = chunk[chunk[self.ID_COL].isin(sample_ids)]
                chunk.to_csv(fp, sep=self.SEP, header=write_header, index=False)
                write_header = False

        self.logger.info("wrote %s" % out_path)


class Learn(Job):

    """runs pyprophet learner on data files in WORKING_FOLDER.
    writes weights files in this folder.
    """

    requires = Subsample
    command_name = "learn"
    options = [working_folder, random_seed]

    def run(self):
        if self.random_seed:
            self.logger.info("set random seed to %r" % self.random_seed)
            random.seed(self.random_seed)

        pathes = tools.scan_files(self.working_folder, filter_by=".txt")
        pyprophet = PyProphet()

        self.logger.info("read subsampled files from %s" % self.working_folder)
        for path in pathes:
            self.logger.info("    read %s" % path)
        tables = list(pyprophet.read_tables_iter(pathes, self.SEP))

        self.logger.info("run pyprophet core algorithm")
        result, scorer, weights = pyprophet._learn_and_apply(tables)
        for line in str(result.summary_statistics).split("\n"):
            self.logger.info(line.rstrip())

        self.logger.info("write sum_stat_subsampled.txt")
        with open(os.path.join(self.working_folder, "sum_stat_subsampled.txt"), "w") as fp:
            result.summary_statistics.to_csv(fp, sep=self.SEP, index=False)

        self.logger.info("write full_stat_subsampled.txt")
        with open(os.path.join(self.working_folder, "full_stat_subsampled.txt"), "w") as fp:
            result.final_statistics.to_csv(fp, sep=self.SEP, index=False)

        self.logger.info("write weights.txt")
        with open(os.path.join(self.working_folder, "weights.txt"), "w") as fp:
            for w in weights:
                print(w, file=fp, end=" ")
