# encoding: utf-8
from __future__ import print_function
import subprocess
import pytest
import os
import shutil
import collections
import tempfile

import pandas


@pytest.fixture(scope="session")
def test_data_folder():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data")


@pytest.fixture(scope="session")
def setup(test_data_folder):
    tmpdir = tempfile.mkdtemp()
    data_folder = os.path.join(tmpdir, "data_folder")
    os.makedirs(data_folder)
    working_folder = os.path.join(tmpdir, "working_folder")
    shutil.copy(os.path.join(test_data_folder, "test_data.txt"), data_folder)

    Setup = collections.namedtuple("Setup", "data_folder working_folder")
    return Setup(data_folder, working_folder)


def test_check(setup):
    cmd = ("pyprophet-brutus check --data-folder %s" % setup.data_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0


def test_subsample(setup):
    cmd = ("pyprophet-brutus subsample --run-local 1 --job-number 1 --job-count 1 "
           "--sample-factor 0.1 "
           "--random-seed 43 "
           "--data-folder %s --working-folder %s") % (setup.data_folder, setup.working_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
    files = os.listdir(setup.working_folder)
    assert len(files) == 1
    assert files[0] == "subsampled_test_data.txt"

    subsamples = pandas.read_csv(os.path.join(setup.working_folder, files[0]), sep="\t")
    assert subsamples.shape == (934, 20)


def test_learn(setup, regtest):
    cmd = ("pyprophet-brutus learn --random-seed 43 --working-folder %s" % setup.working_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0

    files = sorted(os.listdir(setup.working_folder))

    for i, file_ in enumerate(files):
        path = os.path.join(setup.working_folder, file_)
        print(i, "%7d" % os.stat(path).st_size, file_, file=regtest)
