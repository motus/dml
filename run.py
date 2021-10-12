#!/usr/bin/env python3
"""
Starter script to test Deep Mutual Learning loss functions.
NOTE: It requires MPI to run! To launch, do, e.g.:
      `mpiexec -n 4 python3 ./run.py`
"""

import dml
import dml.loss
import numpy as np


def debug_loss(gather, i):
    print("Loss: %d :: [%s]" % (i, ",".join("%s" % g for g in gather)))
    return i


def _main():

    dml_loss = dml.DML(debug_loss)

    for epoch in range(5):
        data = np.ones(3, dtype=np.float32) * (dml_loss.rank + 1)
        loss = dml_loss(data)
        print("Epoch %d Node %d :: %s" % (epoch, dml_loss.rank, loss))

    # Same thing, for batch KL Divergence:

    dml_loss = dml.DML(dml.loss.kl_divergence)

    data = [
        np.array([[9./25, 12./25, 4./25], [9./25, 12./25, 4./25],
                 [1./3, 1./3, 1./3], [1./3, 1./3, 1./3]]),
        np.array([[9./25, 12./25, 4./25], [1./3, 1./3, 1./3],
                 [9./25, 12./25, 4./25], [1./3, 1./3, 1./3]])
    ]

    loss = dml_loss(data[dml_loss.rank % 2])
    print("Result: %d :: %s" % (dml_loss.rank, loss))


if __name__ == "__main__":
    _main()
