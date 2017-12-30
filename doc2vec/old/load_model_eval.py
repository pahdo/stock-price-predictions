from gensim.models import Doc2Vec
from multiprocessing import Pool
import smart_open
import os.path
import time
import glob
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from contextlib import contextmanager
from collections import defaultdict
from collections import OrderedDict
from collections import namedtuple
from gensim.models import Doc2Vec
from IPython.display import HTML
from timeit import default_timer
import gensim.models.doc2vec
import multiprocessing
from os import remove
import numpy as np
import itertools
import datetime
import locale
import gensim
import sys
import re
from sklearn import linear_model
import fileinput
from sklearn.utils import shuffle
from sklearn import ensemble
from sklearn import svm
from sklearn.utils import resample

dirname = 'data'

models = [Doc2Vec.load('saved_doc2vec_models/Doc2Vec(dbow,d100,n5,mc2,s0.001,t8)'), Doc2Vec.load('saved_doc2vec_models/Doc2Vec(dmc,d100,n5,w5,mc2,s0.001,t8)'), Doc2Vec.load('saved_doc2vec_models/Doc2Vec(dmm,d100,n5,w10,mc2,s0.001,t8)')]

for model in models:
    print(model)

classifiers = [linear_model.LogisticRegression(C=1e5), ensemble.RandomForestClassifier(), svm.SVC()]

inferreds = [True, False]

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
num_lines_test = file_len(os.path.join(dirname, 'test-pos.txt'))
num_lines_test += file_len(os.path.join(dirname, 'test-neg.txt'))

def error_rate_for_model(test_model, train_set, test_set, classifier, inferred):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""
    
    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_targets, train_regressors = shuffle(train_targets, train_regressors)
    classifier.fit(train_regressors, train_targets)
    
    if inferred:
        infer_subsample = 0.1
        infer_steps = 3
        infer_alpha = 0.1
        test_targets, test_regressors = zip(*[(doc.sentiment, test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha)) for doc in test_set])
        if infer_subsample < 1.0:
            test_targets, test_regressors = resample(test_targets, test_regressors, n_samples = int(infer_subsample * num_lines_test))
    else:
        test_targets, test_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in test_set])    
        test_targets, test_regressors = shuffle(test_targets, test_regressors)    
    
    # Predict & evaluate
    test_predictions = classifier.predict(test_regressors)
    len_predictions = len(test_predictions)
    corrects = sum(np.rint(test_predictions) == test_targets)
    errors = len_predictions - corrects
    error_rate = float(errors) / len_predictions
    return (error_rate, errors, len_predictions, classifier, inferred)

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
def read_labeled_corpus(fpos, fneg, split):
    f_list = [fpos, fneg]
    i = 0
    for f in f_list:
        if i == 0:
            sentiment = 1.0
        else:
            sentiment = 0.0
        for line in open(f, encoding='utf-8'):
            tokens = gensim.utils.to_unicode(line).split()
            if(len(tokens)==0):
                continue
            words = tokens[1:]
            tags = [tokens[0]]
            yield SentimentDocument(gensim.utils.to_unicode(line).split()[1:], tags, split, sentiment)
        i+=1

results = []
for model in models:
    for classifier in classifiers:
        for inferred in inferreds:
            train_docs = read_labeled_corpus(os.path.join(dirname, 'train-pos.txt'), os.path.join(dirname, 'train-neg.txt'), 'train')
            test_docs = read_labeled_corpus(os.path.join(dirname, 'test-pos.txt'), os.path.join(dirname, 'test-neg.txt'), 'test')
            err, err_count, test_count, predictor, infer = error_rate_for_model(model, train_docs, test_docs, classifier, inferred)
            results.append((err, err_count, test_count, predictor, infer, model))
            print("Error: {0}; Error Count: {1}; Test Count {2}; Predictor: {3}; Inferred: {4}; Model {5}".format(err, err_count, test_count, predictor, infer, model))    