# encoding: utf-8
from __future__ import print_function
import pdb
import subprocess
import pytest
import os
import shutil

import pandas


@pytest.fixture
def test_data_folder():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data")



def test_0(tmpdir):
    out = tmpdir.join("log.txt").strpath
    cmd = ["pyprophet-brutus check --log-level=debug --log-file=%s" % out]
    ret_code = subprocess.call(cmd, shell=True)
    lines = open(out, "r").readlines()
    assert len(lines) == 1


def test_subsample(tmpdir, test_data_folder):
    """
    setup: - data files
           - create temp datafolder + workingfolder
           - copy data file to datafolder
           - run command
           - check working folder

    cmdline:  pyprophet-brutus subsample --job-number 1 --job-count 1 --data-folder XXX
    --working-folder YYY --sample-factor 0.1
    """
    working_folder = tmpdir.join("working_folder").strpath
    data_folder = tmpdir.join("data_folder").mkdir().strpath
    shutil.copy(os.path.join(test_data_folder, "test_data.txt"), data_folder)
    cmd = ("pyprophet-brutus subsample --run-local 1 --job-number 1 --job-count 1 "
           "--sample-factor 0.01 "
           "--random-seed 43 "
           "--data-folder %s --working-folder %s") % (data_folder, working_folder)
    ret_code = subprocess.call(cmd, shell=True)
    files = os.listdir(working_folder)
    assert len(files) == 1

    subsamples = pandas.read_csv(os.path.join(working_folder, files[0]), sep="\t")
    assert subsamples.shape == (120, 20)
