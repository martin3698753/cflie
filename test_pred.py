import batpred as bp
import numpy as np
import maketab as mt


def normalize(train_data):
    mins = np.min(train_data, axis=1, keepdims=True)
    maxs = np.max(train_data, axis=1, keepdims=True)
    normalized_train = (train_data - mins) / (maxs - mins)
    params = {'mins': mins, 'maxs': maxs}
    return normalized_train, params

def denormalize(normalized_train, params):
    mins = params['mins']
    maxs = params['maxs']
    denormalized_train = normalized_train * (maxs - mins) + mins
    return denormalized_train

def load_data(path_dir):
    t, signal = mt.battery(path_dir)
    tleft = 1 - t / max(t)
    #t = np.arange(0, len(tleft), 100)
    return signal, tleft, t

train_data = load_data('data/31-1-25/')
test_data = load_data('data/21-2-25/')
train_data, train_param = normalize(train_data)
test_data, test_param = normalize(test_data)

x = train_data[0][3000:3030]
t = train_data[2][3000:3030]
print("predicting")
bp.eval(x, t)
