import numpy as np
import maketab as mt
from batpred import BatSeqModel

model = BatSeqModel()

def load_data(path_dir):
    t, u = mt.battery(path_dir)
    m = mt.power(path_dir)
    return u, m

u, m = load_data('data/31-1-25/')

for ui, mi in zip(u, m):
    model.pred(ui, mi)

model.done()
