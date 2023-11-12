import pytest
from hbutils.system import TemporaryDirectory

from sdeval.fidelity.ccip import CCIPMetrics
from test.testings import get_testfile


@pytest.fixture(scope='session')
def ccip_init():
    return get_testfile('ccip', 'init')


@pytest.fixture()
def ccip_x():
    return get_testfile('ccip', 'x')


@pytest.fixture()
def ccip_raw():
    return get_testfile('ccip', 'raw')


@pytest.fixture()
def ccip_trained():
    return get_testfile('ccip', 'trained')


@pytest.fixture()
def empty_dir():
    with TemporaryDirectory() as td:
        yield td


@pytest.fixture(scope='session')
def ccip_metric(ccip_init):
    return CCIPMetrics(ccip_init)


@pytest.mark.unittest
class TestFidelityCCIP:
    def test_diff_score(self, ccip_metric, ccip_init, ccip_x, ccip_raw, ccip_trained):
        diff, score = ccip_metric.diff_score(ccip_init)
        assert diff < 0.1
        assert score >= 0.95

        diff, score = ccip_metric.diff_score(ccip_x)
        assert diff >= 0.3
        assert score <= 0.05

        diff, score = ccip_metric.diff_score(ccip_raw)
        assert diff >= 0.05
        assert score < 0.9

        diff, score = ccip_metric.diff_score(ccip_trained)
        assert diff < 0.1
        assert score >= 0.95

    def test_diff(self, ccip_metric, ccip_init, ccip_x, ccip_raw, ccip_trained):
        diff = ccip_metric.diff(ccip_init)
        assert diff < 0.1

        diff = ccip_metric.diff(ccip_x)
        assert diff >= 0.3

        diff = ccip_metric.diff(ccip_raw)
        assert diff >= 0.05

        diff = ccip_metric.diff(ccip_trained)
        assert diff < 0.1

    def test_score(self, ccip_metric, ccip_init, ccip_x, ccip_raw, ccip_trained):
        score = ccip_metric.score(ccip_init)
        assert score >= 0.95

        score = ccip_metric.score(ccip_x)
        assert score <= 0.05

        score = ccip_metric.score(ccip_raw)
        assert score < 0.9

        score = ccip_metric.score(ccip_trained)
        assert score >= 0.95

    def test_failed_init(self, empty_dir):
        with pytest.raises(FileNotFoundError):
            CCIPMetrics(empty_dir)

    def test_failed_calc(self, empty_dir, ccip_metric):
        with pytest.raises(FileNotFoundError):
            ccip_metric.diff_score(empty_dir)
