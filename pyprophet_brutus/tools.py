# encoding: utf-8
from __future__ import print_function

import os


def scan_files(folder):
    input_file_pathes = []
    file_names = os.listdir(folder)
    for file_name in file_names:
        path = os.path.join(folder, file_name)
        if os.path.isfile(path):
            input_file_pathes.append(path)
    return input_file_pathes
