import numpy as np
import maketab as mt
from batpred_seq import BatSeqModel
import matplotlib.pyplot as plt

def load_data(path_dir):
    t, signal = mt.battery(path_dir)
    tleft = 1 - t / max(t)
    return signal, tleft, t

model = BatSeqModel()

signal, _, _ = load_data('data/31-1-25/')
for i in signal[1000:]:
    model.pred(i)

model.done()
