# encoding: utf-8
from __future__ import print_function

import collections
import glob
import os
import shutil
import subprocess
import tempfile

import pandas
import pytest

pandas.options.display.precision = 6
pandas.options.display.width = 200

join = os.path.join


@pytest.fixture(scope="session")
def test_data_folder():
    here = os.path.dirname(os.path.abspath(__file__))
    return join(here, "data")


@pytest.fixture(scope="function", params=[1, 3])
def setup(test_data_folder, request, tmpdir):

    data_folder = tmpdir.join("data_folder")
    data_folder.mkdir()
    work_folder = tmpdir.join("work_folder")

    data_folder = data_folder.strpath
    work_folder = work_folder.strpath

    num_files = request.param
    for i in range(num_files):
        shutil.copy(join(test_data_folder, "test_data.txt"),
                    join(data_folder, "data_%d.txt" % i))

    result_folder = tmpdir.join("result_folder_%d" % num_files).strpath

    Setup = collections.namedtuple(
        "Setup", "data_folder result_folder, work_folder number_input_files")
    return Setup(data_folder, result_folder, work_folder, num_files)


def _prepare(setup, regtest):
    cmd = ("pyprophet-cli prepare --extra-group-column peptide_id --data-folder %s "
           "--work-folder %s" % (setup.data_folder, setup.work_folder))
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0


def _subsample_0(setup, regtest):
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
    subsamples = pandas.read_csv(join(setup.work_folder, files[-1]), sep="\t")
    assert subsamples.shape == (4595, 22)


def _subsample_1(setup, regtest):
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
    subsamples = pandas.read_csv(join(setup.work_folder, files[-1]), sep="\t")
    assert subsamples.shape == (4595, 22)


def _subsample_2(setup, regtest):

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

    subsamples = pandas.read_csv(join(setup.work_folder, files[-1]), sep="\t")
    assert subsamples.shape == (4595, 22)


def ls(folder, regtest):
    files = sorted(os.listdir(folder))
    base_folder = os.path.basename(folder)
    for i, file_ in enumerate(files):
        path = join(folder, file_)
        base_path = join(base_folder, file_)
        print("%2d" % i, "%7d" % os.stat(path).st_size, base_path, file=regtest)


def _learn(setup, regtest):
    cmd = ("pyprophet-cli learn --ignore-invalid-scores --random-seed 43 --work-folder %s"
           % setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0

    ls(setup.work_folder, regtest)

    df = pandas.read_csv(join(setup.work_folder, "weights.txt"), header=None, sep="\t")
    print(file=regtest)
    print("weights", file=regtest)
    df.to_string(regtest)
    print(file=regtest)
    df = pandas.read_csv(join(setup.work_folder, "sum_stat_subsampled.txt"), sep="\t")
    df.to_string(regtest)


def _apply_weights(setup, regtest):
    cmd = ("pyprophet-cli apply_weights --job-number 1 --job-count 1 "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s --work-folder %s") % (setup.data_folder, setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0

    ls(setup.work_folder, regtest)


def _score_pfdr_global(setup, regtest):
    cmd = ("pyprophet-cli score --job-number 1 --job-count 1 "
           "--statistics-mode global "
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
    print(open(join(setup.result_folder, "summary_stats_grouped_by_peptide_id.txt")).read(), file=regtest)
    print(file=regtest)
    print(open(join(
        setup.result_folder, "summary_stats_grouped_by_peptide_id.txt")).read(), file=regtest)

    for i in range(setup.number_input_files):
        with open(join(setup.result_folder, "data_%d_scored.txt" % i)) as fp:
            print(file=regtest)
            print(fp.next(), file=regtest)
            print(fp.next(), file=regtest)


def _score_pfdr_run(setup, regtest):
    cmd = ("pyprophet-cli score --job-number 1 --job-count 1 "
           "--statistics-mode run-specific "
           "--local-folder %s "
           "--overwrite-results "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s "
           "--result-folder %s "
           "--work-folder %s") % (tempfile.mkdtemp(), setup.data_folder, setup.result_folder, setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0

    ls(setup.result_folder, regtest)
    ls(setup.work_folder, regtest)

    for path in glob.glob(join(setup.result_folder, "summary_stats*.txt")):
        print(file=regtest)
        print(os.path.basename(path), file=regtest)
        print(file=regtest)
        print(open(path).read(), file=regtest)

    for i in range(setup.number_input_files):
        with open(join(setup.result_folder, "data_%d_scored.txt" % i)) as fp:
            print(file=regtest)
            print(fp.next(), file=regtest)
            print(fp.next(), file=regtest)


def _score_pfdr_experiment(setup, regtest):
    cmd = ("pyprophet-cli score --job-number 1 --job-count 1 "
           "--statistics-mode experiment-wide "
           "--overwrite-results "
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
    print(open(join(setup.result_folder, "summary_stats_grouped_by_transition_group_id.txt")).read(), file=regtest)
    print(file=regtest)
    print(open(join(
        setup.result_folder, "summary_stats_grouped_by_peptide_id.txt")).read(), file=regtest)

    for i in range(setup.number_input_files):
        with open(join(setup.result_folder, "data_%d_scored.txt" % i)) as fp:
            print(file=regtest)
            print(fp.next(), file=regtest)
            print(fp.next(), file=regtest)


def _score_fdr(setup, regtest):
    cmd = ("pyprophet-cli score --job-number 1 --job-count 1 "
           "--statistics-mode global "
           "--local-folder %s "
           "--data-filename-pattern '*.txt' "
           "--data-folder %s "
           "--use-fdr "
           "--overwrite-results "     # we run scorer a second time and write to resultfolder again
           "--result-folder %s "
           "--work-folder %s") % (tempfile.mkdtemp(), setup.data_folder, setup.result_folder, setup.work_folder)
    ret_code = subprocess.call(cmd, shell=True)
    assert ret_code == 0

    ls(setup.result_folder, regtest)
    ls(setup.work_folder, regtest)

    print(file=regtest)
    print(open(join(setup.result_folder, "summary_stats_grouped_by_transition_group_id.txt")).read(), file=regtest)
    print(file=regtest)
    print(open(join(
        setup.result_folder, "summary_stats_grouped_by_peptide_id.txt")).read(), file=regtest)

    for i in range(setup.number_input_files):
        with open(join(setup.result_folder, "data_%d_scored.txt" % i)) as fp:
            print(file=regtest)
            print(fp.next(), file=regtest)
            print(fp.next(), file=regtest)


def test_global_scoring_variant_0(setup, regtest):
    _prepare(setup, regtest)
    _subsample_0(setup, regtest)
    _learn(setup, regtest)
    _apply_weights(setup, regtest)
    _score_pfdr_global(setup, regtest)


def test_run_scoring(setup, regtest):
    _prepare(setup, regtest)
    _subsample_0(setup, regtest)
    _learn(setup, regtest)
    _apply_weights(setup, regtest)
    _score_pfdr_run(setup, regtest)


def test_experiment_scoring(setup, regtest):
    _prepare(setup, regtest)
    _subsample_0(setup, regtest)
    _learn(setup, regtest)
    _apply_weights(setup, regtest)
    _score_pfdr_experiment(setup, regtest)


def test_global_scoring_variant_1(setup, regtest):
    _prepare(setup, regtest)
    _subsample_1(setup, regtest)
    _learn(setup, regtest)
    _apply_weights(setup, regtest)
    _score_pfdr_global(setup, regtest)


def test_global_scoring_variant_2(setup, regtest):
    _prepare(setup, regtest)
    _subsample_2(setup, regtest)
    _learn(setup, regtest)
    _apply_weights(setup, regtest)
    _score_pfdr_global(setup, regtest)


def test_global_scoring_fdr(setup, regtest):
    _prepare(setup, regtest)
    _subsample_0(setup, regtest)
    _learn(setup, regtest)
    _apply_weights(setup, regtest)
    _score_fdr(setup, regtest)
