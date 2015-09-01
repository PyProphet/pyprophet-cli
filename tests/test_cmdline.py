# encoding: utf-8
from __future__ import print_function
import pdb
import subprocess
import pytest
import os
import shutil
import collections
import tempfile

import pandas

pandas.options.display.precision=6
pandas.options.display.width=200


@pytest.fixture(scope="session")
def test_data_folder():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data")


@pytest.fixture(scope="session", params=[1, 3])
def setup(test_data_folder, request):
    num_files = request.param
    tmpdir = tempfile.mkdtemp()
    data_folder = os.path.join(tmpdir, "data_folder")
    os.makedirs(data_folder)
    working_folder = os.path.join(tmpdir, "working_folder")
    for i in range(num_files):
        shutil.copy(os.path.join(test_data_folder, "test_data.txt"), os.path.join(data_folder, "data_%d.txt" % i))

    Setup = collections.namedtuple("Setup", "data_folder working_folder number_input_files")
    return Setup(data_folder, working_folder, num_files)


def test_check(setup):
    cmd = ("pyprophet-brutus check --data-folder %s" % setup.data_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0


def test_subsample(setup):
    cmd = ("pyprophet-brutus subsample --job-number 1 --job-count 1 "
           "--sample-factor 0.1 "
           "--random-seed 43 "
           "--ignore-invalid-scores "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --working-folder %s") % (setup.data_folder, setup.working_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
    files = os.listdir(setup.working_folder)
    assert len(files) == setup.number_input_files + 1

    files.sort()
    subsamples = pandas.read_csv(os.path.join(setup.working_folder, files[-1]), sep="\t")
    assert subsamples.shape == (934, 21)

    cmd = ("pyprophet-brutus subsample --job-number 1 --job-count 2 "
           "--sample-factor 0.1 "
           "--random-seed 43 "
           "--ignore-invalid-scores "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --working-folder %s "
           "--local-folder %s") % (setup.data_folder, setup.working_folder, tempfile.mkdtemp())
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
    files = os.listdir(setup.working_folder)
    assert len(files) == 2 if setup.number_input_files == 1 else 3

    files.sort()
    subsamples = pandas.read_csv(os.path.join(setup.working_folder, files[-1]), sep="\t")
    assert subsamples.shape == (934, 21)

    cmd = ("pyprophet-brutus subsample --job-number 1 --job-count 2 "
           "--sample-factor 0.1 "
           "--random-seed 43 "
           "--ignore-invalid-scores "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --working-folder %s") % (setup.data_folder, setup.working_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
    files = os.listdir(setup.working_folder)
    assert len(files) == 2 if setup.number_input_files == 1 else 3

    files.sort()

    subsamples = pandas.read_csv(os.path.join(setup.working_folder, files[-1]), sep="\t")
    assert subsamples.shape == (934, 21)


def test_learn(setup, regtest):
    cmd = ("pyprophet-brutus learn --random-seed 43 --working-folder %s" % setup.working_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0

    files = sorted(os.listdir(setup.working_folder))

    for i, file_ in enumerate(files):
        path = os.path.join(setup.working_folder, file_)
        print(i, "%7d" % os.stat(path).st_size, file_, file=regtest)

    for name in ("weights.txt", "sum_stat_subsampled.txt"):
        df = pandas.read_csv(os.path.join(setup.working_folder, name), header=None, sep="\t")
        print(df, file=regtest)
        # print(open(os.path.join(setup.working_folder, name), "r").read(), file=regtest)


def test_score(setup, regtest):
    cmd = ("pyprophet-brutus apply_weights --job-number 1 --job-count 1 "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --working-folder %s") % (setup.data_folder, setup.working_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
