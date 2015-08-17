# encoding: utf-8
from __future__ import print_function

import os



def scan_files(folder, filter_by=None):
    input_file_pathes = []
    file_names = os.listdir(folder)
    for file_name in file_names:
        # TODO: fnmatch instaed endswith:
        if isinstance(filter_by, basestring):
            if not file_name.endswith(filter_by):
                continue
        path = os.path.join(folder, file_name)
        if os.path.isfile(path):
            input_file_pathes.append(path)
    return input_file_pathes
