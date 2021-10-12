# pylint: disable=missing-docstring

import pytest
import numpy as np

import dml.loss


def test_dot_product_vec2():
    gather = [np.array([9./25, 12./25, 4./25]), np.array([1./3, 1./3, 1./3])]
    assert dml.loss.dot_product(gather, 0) == pytest.approx(1./3)
    assert dml.loss.dot_product(gather, 1) == pytest.approx(1./3)


def test_dot_product_vec4():
    gather = [
        np.array([9./25, 12./25, 4./25]), np.array([9./25, 12./25, 4./25]),
        np.array([1./3, 1./3, 1./3]), np.array([1./3, 1./3, 1./3])
    ]
    assert dml.loss.dot_product(gather, 0) == pytest.approx(0.3507556)
    assert dml.loss.dot_product(gather, 1) == pytest.approx(0.3507556)
    assert dml.loss.dot_product(gather, 2) == pytest.approx(1./3)
    assert dml.loss.dot_product(gather, 3) == pytest.approx(1./3)


def test_dot_product_matrix():
    gather = [
        np.array([[9./25, 12./25, 4./25], [9./25, 12./25, 4./25],
                 [1./3, 1./3, 1./3], [1./3, 1./3, 1./3]]),
        np.array([[9./25, 12./25, 4./25], [1./3, 1./3, 1./3],
                 [9./25, 12./25, 4./25], [1./3, 1./3, 1./3]])
    ]
    assert dml.loss.dot_product(gather, 0) == pytest.approx(
        np.array([0.3856, 1./3, 1./3, 1./3]))
    assert dml.loss.dot_product(gather, 1) == pytest.approx(
        np.array([0.3856, 1./3, 1./3, 1./3]))
