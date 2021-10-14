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


def dml_loss(loss_func):
    """
    A decorator that produces a DML loss function applying `loss_func`
    for each element `i` against all other elements in `gather` and
    averaging out the results.
    """
    def _impl(gather, i):
        p = gather[i]
        return sum(loss_func(p, q) for (j, q) in enumerate(gather)
                   if j != i) / (len(gather) - 1)
    return _impl


@dml_loss
def kl_divergence(p, q):
    """
    KL Divergence loss. An average of KL divergencies
    of vector `i` and all other vectors in `gather`.

    gather: a list of NN output from all nodes.
    i: index of the current node.

    returns the average value of a KL Divergence of data `i`
    against all other values in the `gather` list.
    """
    return np.sum(p * np.log(p / q), axis=len(p.shape) - 1)


@dml_loss
def dot_product(p, q):
    """
    Dot Product loss. An average of row-wise dot products
    of `i` and all other vectors or matrices in `gather`.

    gather: a list of NN output from all nodes.
    i: index of the current node.

    returns the average value of row-wise dot products
    of data `i` against all other values in the `gather` list.
    """
    return (p * q).sum(axis=len(p.shape) - 1)


@dml_loss
def cosine_similarity(p, q):
    """
    Cosine Similatity loss. An average of row-wise cosine similarity
    values of `i` and all other vectors or matrices in `gather`.

    gather: a list of NN output from all nodes.
    i: index of the current node.

    returns the average value of row-wise cosine similarities
    of data `i` against all other values in the `gather` list.
    """
    axis = len(p.shape) - 1
    return (p * q).sum(axis=axis) / (
        np.linalg.norm(p, axis=axis) * np.linalg.norm(q, axis=axis))
