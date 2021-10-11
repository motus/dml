#!/usr/bin/env python3

# import numpy
import dml


def debug_loss(gather, i):
    print("%d :: %s" % (i, gather))
    return 0


dml_loss = dml.DML(debug_loss)

data = list(range(dml_loss.rank * 100, dml_loss.rank * 100 + 4))
loss = dml_loss(data)
print(dml_loss.rank, loss)
