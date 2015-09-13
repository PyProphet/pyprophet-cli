# encoding: utf-8
from __future__ import print_function

import os
import fnmatch

import numpy as np

from .exceptions import WorkflowError
from .constants import SCORE_COLUMNS_FILE, INVALID_COLUMNS_FILE


join = os.path.join
basename = os.path.basename
splitext = os.path.splitext


def scan_files(folder, data_filename_pattern=None):
    input_file_pathes = []
    file_names = os.listdir(folder)
    for file_name in file_names:
        if data_filename_pattern is not None and not fnmatch.fnmatch(file_name, data_filename_pattern):
            continue
        path = join(folder, file_name)
        if os.path.isfile(path):
            input_file_pathes.append(path)
    return input_file_pathes


def setup_input_files(job, data_filename_pattern):
    if job.local_folder:
        if not os.path.exists(job.local_folder):
            raise WorkflowError("%s does not exist" % job.local_folder)
    job.input_file_pathes = scan_files(job.data_folder, data_filename_pattern)
    if not job.input_file_pathes:
        raise WorkflowError("data folder %s is empty" % job.data_folder)

    job._pathes_of_files_for_processing = []
    for path in job.input_file_pathes:
        name = basename(path)
        folder = job.local_folder if job.local_folder else job.data_folder
        path = join(folder, name)
        job._pathes_of_files_for_processing.append(path)


def file_name_stem(path):
    return splitext(basename(path))[0]


def setup_dtypes(work_folder):
    dtype = {}
    with open(join(work_folder, SCORE_COLUMNS_FILE), "r") as fp:
        for line in fp:
            dtype[line.rstrip()] = np.float32
    return dtype


def read_column_names(folder, file_name):
    with open(os.path.join(folder, file_name), "r") as fp:
        return [line.rstrip() for line in fp]


def write_column_names(names, folder, file_name):
    with open(os.path.join(folder, file_name), "w") as fp:
        for name in names:
            print(name, file=fp)
