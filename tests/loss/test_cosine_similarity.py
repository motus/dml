# pylint: disable=missing-docstring

import pytest
import numpy as np

import dml.loss


def test_cosine_similarity_vec2():
    gather = [np.array([9./25, 12./25, 4./25]), np.array([1./3, 1./3, 1./3])]
    assert dml.loss.cosine_similarity(gather, 0) == pytest.approx(0.92976)
    assert dml.loss.cosine_similarity(gather, 1) == pytest.approx(0.92976)


def test_cosine_similarity_vec4():
    gather = [
        np.array([9./25, 12./25, 4./25]), np.array([9./25, 12./25, 4./25]),
        np.array([1./3, 1./3, 1./3]), np.array([1./3, 1./3, 1./3])
    ]
    assert dml.loss.cosine_similarity(gather, 0) == pytest.approx(0.953173336)
    assert dml.loss.cosine_similarity(gather, 1) == pytest.approx(0.953173336)
    assert dml.loss.cosine_similarity(gather, 2) == pytest.approx(0.953173336)
    assert dml.loss.cosine_similarity(gather, 3) == pytest.approx(0.953173336)


def test_cosine_similarity_matrix():
    gather = [
        np.array([[9./25, 12./25, 4./25], [9./25, 12./25, 4./25],
                 [1./3, 1./3, 1./3], [1./3, 1./3, 1./3]]),
        np.array([[9./25, 12./25, 4./25], [1./3, 1./3, 1./3],
                 [9./25, 12./25, 4./25], [1./3, 1./3, 1./3]])
    ]
    assert dml.loss.cosine_similarity(gather, 0) == pytest.approx(
        np.array([1, 0.92976, 0.92976, 1]))
    assert dml.loss.cosine_similarity(gather, 1) == pytest.approx(
        np.array([1, 0.92976, 0.92976, 1]))
