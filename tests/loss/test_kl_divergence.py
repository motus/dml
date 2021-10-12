# pylint: disable=missing-docstring

import pytest
import numpy as np

import dml.loss


def test_kl_divergence_loss2():
    gather = [np.array([9./25, 12./25, 4./25]), np.array([1./3, 1./3, 1./3])]
    assert dml.loss.kl_divergence(gather, 0) == pytest.approx(0.0852996)
    assert dml.loss.kl_divergence(gather, 1) == pytest.approx(0.097455)


def test_kl_divergence_loss4():
    gather = [
        np.array([9./25, 12./25, 4./25]), np.array([9./25, 12./25, 4./25]),
        np.array([1./3, 1./3, 1./3]), np.array([1./3, 1./3, 1./3])
    ]
    assert dml.loss.kl_divergence(gather, 0) == pytest.approx(0.0568664)
    assert dml.loss.kl_divergence(gather, 1) == pytest.approx(0.0568664)
    assert dml.loss.kl_divergence(gather, 2) == pytest.approx(0.06497)
    assert dml.loss.kl_divergence(gather, 3) == pytest.approx(0.06497)


def test_kl_divergence_matrix():
    gather = [
        np.array([[9./25, 12./25, 4./25], [9./25, 12./25, 4./25],
                 [1./3, 1./3, 1./3], [1./3, 1./3, 1./3]]),
        np.array([[9./25, 12./25, 4./25], [1./3, 1./3, 1./3],
                 [9./25, 12./25, 4./25], [1./3, 1./3, 1./3]])
    ]
    assert dml.loss.kl_divergence(gather, 0) == pytest.approx(
        np.array([0, 0.0852996, 0.097455, 0]))
    assert dml.loss.kl_divergence(gather, 1) == pytest.approx(
        np.array([0, 0.097455, 0.0852996, 0]))
