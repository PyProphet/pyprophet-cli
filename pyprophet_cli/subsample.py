# encoding: utf-8
from __future__ import print_function

import collections
import os
import random
import shutil

import pandas as pd

import core
from common_options import (job_number, job_count, local_folder, separator, data_folder,
                            work_folder, random_seed, chunk_size, sample_factor,
                            data_filename_pattern)

import io

from constants import SUBSAMPLED_FILES_PATTERN
from exceptions import InvalidInput


class Subsample(core.Job):

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
        io.setup_input_files(self, self.data_filename_pattern)

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
        stem = io.file_name_stem(path)
        if self.local_folder:
            out_path = os.path.join(self.local_folder, SUBSAMPLED_FILES_PATTERN % stem)
        else:
            out_path = os.path.join(self.work_folder, SUBSAMPLED_FILES_PATTERN % stem)
        self.logger.info("   start to subsample from %s and write to %s" % (path, out_path))

        dtype = io.setup_dtypes(self.work_folder)
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

