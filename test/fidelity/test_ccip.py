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
    def test_score(self, ccip_metric, ccip_init, ccip_x, ccip_raw, ccip_trained):
        assert ccip_metric.score(ccip_init) >= 0.95
        assert ccip_metric.score(ccip_x) <= 0.05
        assert ccip_metric.score(ccip_raw) < 0.9
        assert ccip_metric.score(ccip_trained) >= 0.95

    def test_failed_init(self, empty_dir):
        with pytest.raises(FileNotFoundError):
            CCIPMetrics(empty_dir)

    def test_failed_calc(self, empty_dir, ccip_metric):
        with pytest.raises(FileNotFoundError):
            ccip_metric.score(empty_dir)
