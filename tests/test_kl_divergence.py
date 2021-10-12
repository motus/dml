# pylint: disable=missing-docstring

import pytest
import numpy as np

import dml


def test_kl_divergence_func():
    p = np.array([9, 12, 4], dtype=np.float) / 25.0
    q = np.ones(3, dtype=np.float) / 3.0
    assert dml.dml._kl_divergence(p, q) == pytest.approx(0.0852996)
    assert dml.dml._kl_divergence(q, p) == pytest.approx(0.097455)
