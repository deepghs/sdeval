import glob
import os.path

import numpy as np
import pytest
from hbutils.system import TemporaryDirectory

from sdeval.fidelity.ccip import CCIPMetrics
from test.testings import get_testfile


@pytest.fixture(scope='session')
def ccip_init():
    return os.path.relpath(get_testfile('ccip', 'init'), os.path.abspath('.'))


@pytest.fixture()
def ccip_init_files(ccip_init):
    return glob.glob(os.path.join(ccip_init, '*.jpg'))


@pytest.fixture()
def ccip_x():
    return get_testfile('ccip', 'x')


@pytest.fixture()
def ccip_x_files(ccip_x):
    return glob.glob(os.path.join(ccip_x, '*.png'))


@pytest.fixture()
def ccip_raw():
    return get_testfile('ccip', 'raw')


@pytest.fixture()
def ccip_raw_files(ccip_raw):
    return glob.glob(os.path.join(ccip_raw, '*.png'))


@pytest.fixture()
def ccip_trained():
    return get_testfile('ccip', 'trained')


@pytest.fixture()
def ccip_trained_files(ccip_trained):
    return glob.glob(os.path.join(ccip_trained, '*.png'))


@pytest.fixture()
def ccip_amiya_feats_bin():
    return np.load(get_testfile('ccip_amiya', 'features.npy'))


@pytest.fixture()
def ccip_amiya_feats(ccip_amiya_feats_bin):
    return CCIPMetrics(None, feats=ccip_amiya_feats_bin)


@pytest.fixture()
def ccip_amiya_pos():
    return get_testfile('ccip_amiya', 'amiya')


@pytest.fixture()
def ccip_amiya_neg():
    return get_testfile('ccip_amiya', 'non_amiya')


@pytest.fixture()
def empty_dir():
    with TemporaryDirectory() as td:
        yield td


@pytest.fixture(scope='session')
def ccip_metric(ccip_init):
    return CCIPMetrics(ccip_init)


@pytest.mark.unittest
class TestFidelityCCIP:
    def test_score(self, ccip_metric, ccip_init, ccip_x, ccip_raw, ccip_trained):
        assert ccip_metric.score(ccip_init) >= 0.95
        assert ccip_metric.score(ccip_x) <= 0.05
        assert ccip_metric.score(ccip_raw) < 0.9
        assert ccip_metric.score(ccip_trained) >= 0.95

    def test_score_files(self, ccip_metric, ccip_init_files, ccip_x_files, ccip_raw_files, ccip_trained_files):
        assert ccip_metric.score(ccip_init_files) >= 0.95
        assert ccip_metric.score(ccip_x_files) <= 0.05
        assert ccip_metric.score(ccip_raw_files) < 0.9
        assert ccip_metric.score(ccip_trained_files) >= 0.95

    def test_score_files_seq(self, ccip_metric, ccip_init_files, ccip_x_files, ccip_raw_files, ccip_trained_files):
        seq = ccip_metric.score(ccip_init_files, mode='seq')
        assert seq.shape == (len(ccip_init_files),)
        assert seq.mean().item() >= 0.95

        seq = ccip_metric.score(ccip_x_files, mode='seq')
        assert seq.shape == (len(ccip_x_files),)
        assert seq.mean().item() <= 0.05

        seq = ccip_metric.score(ccip_raw_files, mode='seq')
        assert seq.shape == (len(ccip_raw_files),)
        assert seq.mean().item() < 0.9

        seq = ccip_metric.score(ccip_trained_files, mode='seq')
        assert seq.shape == (len(ccip_trained_files),)
        assert seq.mean().item() >= 0.95

    def test_score_amiya(self, ccip_amiya_feats, ccip_amiya_pos, ccip_amiya_neg):
        assert ccip_amiya_feats.score(ccip_amiya_pos) >= 0.65
        assert ccip_amiya_feats.score(ccip_amiya_neg) <= 0.1

    def test_failed_init(self, empty_dir):
        with pytest.raises(FileNotFoundError):
            CCIPMetrics(empty_dir)

    def test_failed_calc(self, empty_dir, ccip_metric):
        with pytest.raises(FileNotFoundError):
            ccip_metric.score(empty_dir)

    def test_failed_images(self, ccip_init, ccip_amiya_feats_bin, ccip_amiya_pos, ccip_amiya_neg,
                           ccip_x, ccip_trained):
        with pytest.warns(UserWarning):
            metrics = CCIPMetrics(ccip_init, feats=ccip_amiya_feats_bin)

        assert metrics.score(ccip_amiya_pos) >= 0.65
        assert metrics.score(ccip_amiya_neg) <= 0.1
        assert metrics.score(ccip_x) <= 0.1
        assert metrics.score(ccip_trained) <= 0.1
