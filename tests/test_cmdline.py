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
    cmd = ("pyprophet-cli prepare --extra-group-column transition_group_id --data-folder %s "
           "--work-folder %s" % (setup.data_folder, setup.work_folder))
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0


def test_subsample(setup):
    cmd = ("pyprophet-cli subsample --job-number 1 --job-count 1 "
           "--sample-factor 0.5 "
           "--random-seed 43 "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --work-folder %s") % (setup.data_folder, setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
    files = os.listdir(setup.work_folder)
    assert len(files) == setup.number_input_files + 2

    files.sort()
    subsamples = pandas.read_csv(os.path.join(setup.work_folder, files[-1]), sep="\t")
    assert subsamples.shape == (4595, 21)

    cmd = ("pyprophet-cli subsample --job-number 1 --job-count 2 "
           "--sample-factor 0.5 "
           "--random-seed 43 "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --work-folder %s "
           "--local-folder %s") % (setup.data_folder, setup.work_folder, tempfile.mkdtemp())
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
    files = os.listdir(setup.work_folder)
    assert len(files) == 3 if setup.number_input_files == 1 else 4

    files.sort()
    subsamples = pandas.read_csv(os.path.join(setup.work_folder, files[-1]), sep="\t")
    assert subsamples.shape == (4595, 21)

    cmd = ("pyprophet-cli subsample --job-number 1 --job-count 2 "
           "--sample-factor 0.5 "
           "--random-seed 43 "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --work-folder %s") % (setup.data_folder, setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0
    files = os.listdir(setup.work_folder)
    assert len(files) == 3 if setup.number_input_files == 1 else 4

    files.sort()

    subsamples = pandas.read_csv(os.path.join(setup.work_folder, files[-1]), sep="\t")
    assert subsamples.shape == (4595, 21)


def ls(folder, regtest):
    files = sorted(os.listdir(folder))
    base_folder = os.path.basename(folder)
    for i, file_ in enumerate(files):
        path = os.path.join(folder, file_)
        base_path = os.path.join(base_folder, file_)
        print("%2d" % i, "%7d" % os.stat(path).st_size, base_path, file=regtest)


def test_learn(setup, regtest):
    cmd = ("pyprophet-cli learn --ignore-invalid-scores --random-seed 43 --work-folder %s"
           % setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0

    ls(setup.work_folder, regtest)

    df = pandas.read_csv(os.path.join(setup.work_folder, "weights.txt"), header=None, sep="\t")
    df.to_string(regtest)
    df = pandas.read_csv(os.path.join(setup.work_folder, "sum_stat_subsampled.txt"), sep="\t")
    df.to_string(regtest)


def test_apply_weights(setup, regtest):
    cmd = ("pyprophet-cli apply_weights --job-number 1 --job-count 1 "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --work-folder %s") % (setup.data_folder, setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0

    ls(setup.work_folder, regtest)


def test_score(setup, regtest):
    cmd = ("pyprophet-cli score --job-number 1 --job-count 1 "
           "--local-folder %s "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s "
           "--result-folder %s "
           "--work-folder %s") % (tempfile.mkdtemp(), setup.data_folder, setup.result_folder, setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0

    ls(setup.result_folder, regtest)
    ls(setup.work_folder, regtest)

    print(file=regtest)
    print(open(os.path.join(setup.result_folder, "summary_stats.txt")).read(), file=regtest)
    print(file=regtest)
    print(open(os.path.join(setup.result_folder, "summary_stats_grouped_by_transition_group_id.txt")).read(), file=regtest)

    for i in range(setup.number_input_files):
        with open(os.path.join(setup.result_folder, "data_%d_scored.txt" % i)) as fp:
            print(file=regtest)
            print(fp.next(), file=regtest)
            print(fp.next(), file=regtest)

