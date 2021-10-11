#!/usr/bin/env python3
"""Deep Mutual Learning library"""

import numpy as np
from mpi4py import MPI


class DML:
    "DML wrapper for a loss function"

    def __init__(self, loss_func):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.loss = loss_func
        self.tag = 0

    def __call__(self, data):
        "Compute the consensus loss function"
        self.tag += 1
        gather = self.comm.gather(data, root=0)
        if self.rank > 0:
            return self.comm.recv(source=0, tag=self.tag)
        for i in range(1, self.size):
            loss = self.loss(gather, i)
            self.comm.send(loss, dest=i, tag=self.tag)
        return self.loss(gather, 0)


def null_loss(gather, i):
    """Always return zero loss.

    gather: a list of NN output from all nodes.
    i: index of the current node.

    returns the value of a loss of data `i`
    against all other values in the `gather` list.
    """
    print("Null loss: node %d: %s" % (i, gather))
    return 0


def _kl_divergence(p, q):
    "KL Divergence of two vectors"
    return np.sum(p * np.log(p / q))


def kl_divergence_loss(gather, i):
    """
    KL Divergence loss. An average of KL divergencies
    of vector `i` and all other vectors in `gather`.

    gather: a list of NN output from all nodes.
    i: index of the current node.

    returns the average value of a KL Divergence of data `i`
    against all other values in the `gather` list.
    """
    p = gather[i]
    return sum(_kl_divergence(p, q) for (j, q) in enumerate(gather)
               if j != i) / (len(gather) - 1)
