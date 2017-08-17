from multiprocessing import Pool
import smart_open
import os.path
import spacy
import time
import glob

nlp = spacy.load('en')
dirname = 'data'

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

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

num_lines_alldata =  file_len(os.path.join(dirname, 'alldata-id.txt'))
num_lines_train =  file_len(os.path.join(dirname, 'alldata-id.txt'))
num_lines_test =  file_len(os.path.join(dirname, 'alldata-id.txt'))
print("Num lines alldata {}".format(num_lines_alldata))

def read_corpus(fname):
    for line in open(fname, encoding='utf-8'):
        # For training data, add tags
        tokens = gensim.utils.to_unicode(line).split()
        if(len(tokens)==0):
            continue
        words = tokens[1:]
        tags = [tokens[0]]
        yield gensim.models.doc2vec.TaggedDocument(' '.join(words), tags)
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

cores = multiprocessing.cpu_count()
print("{} cores".format(cores))
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

simple_models = [
    # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
    # Every 10 million word types need about 1GB of RAM (For setting max_vocab_size)
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, max_vocab_size=100000, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW 
    Doc2Vec(dm=0, size=100, max_vocab_size=100000, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/ average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, max_vocab_size=100000, negative=5, hs=0, min_count=2, workers=cores),
]

#simple_models = [
#    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, max_vocab_size=100000, negative=5, hs=0, min_count=2, workers=cores),
#    Doc2Vec(dm=0, size=100, max_vocab_size=100000, negative=5, hs=0, min_count=2, workers=cores)
#]

# Speed up setup by sharing results of the 1st model's vocabulary scan
print("Building a vocabulary...")
st_time = time.time()
simple_models[0].build_vocab(read_corpus(os.path.join(dirname, 'alldata-id.txt')))  # PV-DM w/ concat requires one special NULL word so it serves as template
end_tm = time.time()
print("Vocabulary built for the first time in {0}".format(end_tm-st_time))
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])

models_by_name = OrderedDict((str(model).replace('/',''), model) for model in simple_models)

models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])
for model in models_by_name:
    print(model)

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

#def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
def error_rate_for_model(test_model, train_set, test_set):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""
    
    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_targets, train_regressors = shuffle(train_targets, train_regressors)
    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(train_regressors, train_targets)
    #train_targets, train_regressors = None, None # Uncommenting these breaks the code

    test_regressors_sentiment = [(test_model.docvecs[doc.tags[0]], doc.sentiment) for doc in test_set]    
    test_regressors = [doc[0] for doc in test_regressors_sentiment]
    answers = [doc[1] for doc in test_regressors_sentiment]
    answers, test_regressors = shuffle(answers, test_regressors)
    print(answers)
    
    # Predict & evaluate
    test_predictions = logistic.predict(train_regressors)
    print(test_predictions)
    len_predictions = len(test_predictions)
    #test_regressors = None # Uncommenting these breaks the code
    corrects = sum(np.rint(test_predictions) == answers)
    #test_predictions = None # Uncommenting these breaks the code
    errors = len_predictions - corrects
    error_rate = float(errors) / len_predictions
    return (error_rate, errors, len_predictions, logistic)
    
# Cell
best_error = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved

alpha, min_alpha, passes = (0.025, 0.001, 5)
alpha_delta = (alpha - min_alpha) / passes

print("START %s" % datetime.datetime.now())
start = time.time()

for epoch in range(passes):
    for name, train_model in models_by_name.items():
        # Note these are generator functions so they can only be read through once    
        doc_list = read_corpus(os.path.join(dirname, 'alldata-id.txt'))
        train_docs = read_labeled_corpus(os.path.join(dirname, 'train-pos.txt'), os.path.join(dirname, 'train-neg.txt'), 'train')
        test_docs = read_labeled_corpus(os.path.join(dirname, 'test-pos.txt'), os.path.join(dirname, 'test-neg.txt'), 'test')

        # Train
        duration = 'na'
        train_model.alpha, train_model.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            train_model.train(doc_list, total_examples=num_lines_alldata, epochs=1)
            duration = '%.1f' % elapsed()
            
        # Evaluate
        eval_duration = ''
        with elapsed_timer() as eval_elapsed:
            err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs, test_docs)
        eval_duration = '%.1f' % eval_elapsed()
        best_indicator = ' '
        if err <= best_error[name]:
            best_error[name] = err
            best_indicator = '*'
        print("%s%f : %i passes : %s %ss %ss" % (best_indicator, err, epoch + 1, name, duration, eval_duration))

        #if (epoch == passes-1):
        #    train_model.save()
            
    print('Completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta

print("END %s" % str(datetime.datetime.now()))
end = time.time()
print("Time elapsed: {0}".format(end-start))

# Cell
# Print best error rates achieved
print("Err rate Model")
for rate, name in sorted((rate, name) for name, rate in best_error.items()):
    print("%f %s" % (rate, name))
    
for train_model in simple_models:
    train_model.save(str(train_model).replace('/',''))
