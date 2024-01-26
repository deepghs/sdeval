import glob
import os.path

import pytest
from hbutils.testing import isolated_directory

from sdeval.corrupt import AICorruptMetrics
from test.testings import get_testfile


@pytest.fixture()
def aicorrupt_o():
    return get_testfile('aicorrupt', 'o')


@pytest.fixture()
def aicorrupt_o_files(aicorrupt_o):
    return glob.glob(os.path.join(aicorrupt_o, '*.png'))


@pytest.fixture()
def aicorrupt_x():
    return get_testfile('aicorrupt', 'x')


@pytest.fixture()
def aicorrupt_x_files(aicorrupt_x):
    return glob.glob(os.path.join(aicorrupt_x, '*.png'))


@pytest.fixture()
def aicorrupt_metrics():
    return AICorruptMetrics()


# noinspection PyCallingNonCallable
@pytest.mark.unittest
class TestCorruptAICorrupt:
    def test_score(self, aicorrupt_o, aicorrupt_x, aicorrupt_metrics):
        assert aicorrupt_metrics.score(aicorrupt_o) >= 0.97
        assert aicorrupt_metrics.score(aicorrupt_x) < 0.05

    def test_score_files(self, aicorrupt_o_files, aicorrupt_x_files, aicorrupt_metrics):
        assert aicorrupt_metrics.score(aicorrupt_o_files) >= 0.97
        assert aicorrupt_metrics.score(aicorrupt_x_files) < 0.05

    def test_score_files_seq(self, aicorrupt_o_files, aicorrupt_x_files, aicorrupt_metrics):
        seq = aicorrupt_metrics.score(aicorrupt_o_files, mode='seq')
        assert seq.shape == (len(aicorrupt_o_files),)
        assert seq.mean().item() >= 0.97

        seq = aicorrupt_metrics.score(aicorrupt_x_files, mode='seq')
        assert seq.shape == (len(aicorrupt_x_files),)
        assert seq.mean().item() < 0.05

    @isolated_directory()
    def test_aicorrupt_failed(self, aicorrupt_metrics):
        with pytest.raises(FileNotFoundError):
            _ = aicorrupt_metrics.score('.')
