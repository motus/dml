# pylint: disable=missing-docstring

import pytest
import numpy as np

import dml


def test_kl_divergence_func():
    p = np.array([9./25, 12./25, 4./25])
    q = np.array([1./3, 1./3, 1./3])
    assert dml.dml._kl_divergence(p, q) == pytest.approx(0.0852996)
    assert dml.dml._kl_divergence(q, p) == pytest.approx(0.097455)
