import pytest
from hbutils.testing import isolated_directory

from sdeval.corrupt import AICorruptMetrics
from test.testings import get_testfile


@pytest.fixture()
def aicorrupt_o():
    return get_testfile('aicorrupt', 'o')


@pytest.fixture()
def aicorrupt_x():
    return get_testfile('aicorrupt', 'x')


@pytest.fixture()
def aicorrupt_metrics():
    return AICorruptMetrics()


# noinspection PyCallingNonCallable
@pytest.mark.unittest
class TestCorruptAICorrupt:
    def test_aicorrupt(self, aicorrupt_o, aicorrupt_x, aicorrupt_metrics):
        assert aicorrupt_metrics.corrupt_score(aicorrupt_o) < 0.03
        assert aicorrupt_metrics.corrupt_score(aicorrupt_x) >= 0.95

    @isolated_directory()
    def test_aicorrupt_failed(self, aicorrupt_metrics):
        with pytest.raises(FileNotFoundError):
            _ = aicorrupt_metrics.corrupt_score('.')
