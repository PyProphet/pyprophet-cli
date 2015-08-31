# encoding: utf-8
from __future__ import print_function

import os
import fnmatch


def scan_files(folder, data_filename_pattern=None):
    input_file_pathes = []
    file_names = os.listdir(folder)
    for file_name in file_names:
        if data_filename_pattern is not None and not fnmatch.fnmatch(file_name, data_filename_pattern):
            continue
        path = os.path.join(folder, file_name)
        if os.path.isfile(path):
            input_file_pathes.append(path)
    return input_file_pathes
