import os
import pickle

data_dir = '../data'
fname = 'api.info'
print('hello world.'.split())
with open(os.path.join(data_dir,fname),'rb') as f:
    print(pickle.load(f))