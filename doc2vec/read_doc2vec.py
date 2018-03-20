import numpy as np
import pickle
import codecs

with open('data/dataset/doc2vec_test0.txt') as f:
    for line in f.readlines():
        v, p, a = line.split('\t')
        v = pickle.loads(codecs.decode(v.encode(), 'base64'))
