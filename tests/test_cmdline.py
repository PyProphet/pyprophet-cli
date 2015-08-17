# encoding: utf-8
from __future__ import print_function
import subprocess
import pytest
import os
import shutil
import collections

import pandas


@pytest.fixture
def test_data_folder():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data")


@pytest.fixture
def setup(tmpdir, test_data_folder):
    data_folder = tmpdir.join("data_folder").mkdir().strpath
    working_folder = tmpdir.join("working_folder").strpath
    shutil.copy(os.path.join(test_data_folder, "test_data.txt"), data_folder)

    Setup = collections.namedtuple("Setup", "data_folder working_folder")
    return Setup(data_folder, working_folder)


def test_check(setup):
    cmd = ("pyprophet-brutus check --data-folder %s" % setup.data_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0


def test_subsample(setup):
    cmd = ("pyprophet-brutus subsample --run-local 1 --job-number 1 --job-count 1 "
           "--sample-factor 0.01 "
           "--random-seed 43 "
           "--data-folder %s --working-folder %s") % (setup.data_folder, setup.working_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
    files = os.listdir(setup.working_folder)
    assert len(files) == 1
    assert files[0] == "subsampled_test_data.txt"

    subsamples = pandas.read_csv(os.path.join(setup.working_folder, files[0]), sep="\t")
    assert subsamples.shape == (120, 20)
