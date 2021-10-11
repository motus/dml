#!/usr/bin/env python3

import dml
import numpy as np


def debug_loss(gather, i):
    print("%d :: %s" % (i, gather))
    return 0


dml_loss = dml.DML(debug_loss)

data = np.ones(3, dtype=np.float32) * (dml_loss.rank + 1)
loss = dml_loss(data)
print(dml_loss.rank, loss)
