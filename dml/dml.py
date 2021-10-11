#!/usr/bin/env python3
"""Deep Mutual Learning library"""

from mpi4py import MPI


class DML:

    def __init__(self, loss_func):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.loss = loss_func
        self.tag = 0

    def __call__(self, data):
        self.tag += 1
        gather = self.comm.gather(data, root=0)
        if self.rank > 0:
            return self.comm.recv(source=0, tag=self.tag)
        for i in range(1, self.size):
            loss = self.loss(data, gather, i)
            self.comm.send(loss, dest=i, tag=self.tag)
        return self.loss(data, gather, 0)
        
