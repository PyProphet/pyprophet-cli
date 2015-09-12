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
    print("----- tmpdir is", tmpdir)
    data_folder = os.path.join(tmpdir, "data_folder")
    os.makedirs(data_folder)
    work_folder = os.path.join(tmpdir, "work_folder")
    for i in range(num_files):
        shutil.copy(os.path.join(test_data_folder, "test_data.txt"),
                    os.path.join(data_folder, "data_%d.txt" % i))

    result_folder = os.path.join(tmpdir, "result_folder_%d" % num_files)

    Setup = collections.namedtuple("Setup", "data_folder result_folder, work_folder number_input_files")
    return Setup(data_folder, result_folder, work_folder, num_files)


def test_prepare(setup):
    cmd = ("pyprophet-cli prepare --data-folder %s --work-folder %s" % (setup.data_folder,
                                                                           setup.work_folder))
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0


def test_subsample(setup):
    cmd = ("pyprophet-cli subsample --job-number 1 --job-count 1 "
           "--sample-factor 0.1 "
           "--random-seed 43 "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --work-folder %s") % (setup.data_folder, setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
    files = os.listdir(setup.work_folder)
    assert len(files) == setup.number_input_files + 1

    files.sort()
    subsamples = pandas.read_csv(os.path.join(setup.work_folder, files[-1]), sep="\t")
    assert subsamples.shape == (934, 21)

    cmd = ("pyprophet-cli subsample --job-number 1 --job-count 2 "
           "--sample-factor 0.1 "
           "--random-seed 43 "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --work-folder %s "
           "--local-folder %s") % (setup.data_folder, setup.work_folder, tempfile.mkdtemp())
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
    files = os.listdir(setup.work_folder)
    assert len(files) == 2 if setup.number_input_files == 1 else 3

    files.sort()
    subsamples = pandas.read_csv(os.path.join(setup.work_folder, files[-1]), sep="\t")
    assert subsamples.shape == (934, 21)

    cmd = ("pyprophet-cli subsample --job-number 1 --job-count 2 "
           "--sample-factor 0.1 "
           "--random-seed 43 "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --work-folder %s") % (setup.data_folder, setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
    files = os.listdir(setup.work_folder)
    assert len(files) == 2 if setup.number_input_files == 1 else 3

    files.sort()

    subsamples = pandas.read_csv(os.path.join(setup.work_folder, files[-1]), sep="\t")
    assert subsamples.shape == (934, 21)


def test_learn(setup, regtest):
    cmd = ("pyprophet-cli learn --ignore-invalid-scores --random-seed 43 --work-folder %s"
            % setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0

    files = sorted(os.listdir(setup.work_folder))

    for i, file_ in enumerate(files):
        path = os.path.join(setup.work_folder, file_)
        print(i, "%7d" % os.stat(path).st_size, file_, file=regtest)

    df = pandas.read_csv(os.path.join(setup.work_folder, "weights.txt"), header=None, sep="\t")
    print(df, file=regtest)
    df = pandas.read_csv(os.path.join(setup.work_folder, "sum_stat_subsampled.txt"), sep="\t")
    print(df, file=regtest)


def test_apply_weights(setup, regtest):
    cmd = ("pyprophet-cli apply_weights --job-number 1 --job-count 1 "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --work-folder %s") % (setup.data_folder, setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0


def test_score(setup, regtest):
    cmd = ("pyprophet-cli score --job-number 1 --job-count 1 "
           "--local-folder %s "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s "
           "--result-folder %s "
           "--work-folder %s") % (tempfile.mkdtemp(), setup.data_folder, setup.result_folder, setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
