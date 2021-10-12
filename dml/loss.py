#!/usr/bin/env python3
"""Loss functions for the Deep Mutual Learning library"""

import numpy as np


def null(gather, i):
    """Always return zero loss.

    gather: a list of NN output from all nodes.
    i: index of the current node.

    returns 0.
    """
    return 0


def _kl_divergence_impl(p, q):
    "KL Divergence of two vectors"
    return np.sum(p * np.log(p / q), axis=len(p.shape) - 1)


def kl_divergence(gather, i):
    """
    KL Divergence loss. An average of KL divergencies
    of vector `i` and all other vectors in `gather`.

    gather: a list of NN output from all nodes.
    i: index of the current node.

    returns the average value of a KL Divergence of data `i`
    against all other values in the `gather` list.
    """
    p = gather[i]
    return sum(_kl_divergence_impl(p, q) for (j, q) in enumerate(gather)
               if j != i) / (len(gather) - 1)
