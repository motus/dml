#!/usr/bin/env python3

import dml
import numpy as np


def debug_loss(gather, i):
    print("Loss: %d :: [%s]" % (i, ",".join("%s" % g for g in gather)))
    return 0


dml_loss = dml.DML(debug_loss)

data = np.ones(3, dtype=np.float32) * (dml_loss.rank + 1)
loss = dml_loss(data)
print("Result: %d :: %s" % (dml_loss.rank, loss))


# Same thing, for batch KL Divergence:

dml_loss = dml.DML(dml.kl_divergence_loss)

data = [
    np.array([[9./25, 12./25, 4./25], [9./25, 12./25, 4./25],
              [1./3, 1./3, 1./3], [1./3, 1./3, 1./3]]),
    np.array([[9./25, 12./25, 4./25], [1./3, 1./3, 1./3],
              [9./25, 12./25, 4./25], [1./3, 1./3, 1./3]])
]

loss = dml_loss(data[dml_loss.rank % 2])
print("Result: %d :: %s" % (dml_loss.rank, loss))
