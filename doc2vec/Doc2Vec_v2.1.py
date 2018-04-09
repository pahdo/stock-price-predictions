from multiprocessing import Pool
import numpy as np
import os.path
import time
from contextlib import contextmanager
from gensim.models import Doc2Vec
from timeit import default_timer
import gensim.models.doc2vec
import multiprocessing
import itertools
import datetime
import gensim
import sys
import fileinput
import utils_v2
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pickle
import custom_transformers

first_model = None

def train_model(Xa, model_type, size, window):
    st = time.time() 
    if model_type == 'dm_mean':
        model = Doc2Vec(dm=1, dm_mean=1, size=size, window=window, max_vocab_size=100000, negative=5, hs=1, min_count=2, workers=cores,)
    elif model_type == 'dm_concat':
        model = Doc2Vec(dm=1, dm_concat=1, size=size, window=window, max_vocab_size=100000, negative=5, hs=1, min_count=2, workers=cores,)
    elif model_type == 'dbow':
        model = Doc2Vec(dm=0, size=size, window=window, max_vocab_size=100000, negative=5, hs=1, min_count=2, workers=cores,)
    if first_model is None:
        print("Building a vocabulary...")
        start_tm = time.time()
        model.build_vocab(Xa)
        end_tm = time.time()
        print("Vocabulary built for the first time in {0}".format(end_tm-start_tm))
    else:
        model.reset_from(first_model)
    print(model)
    
    @contextmanager
    def elapsed_timer():
        start = default_timer()
        elapser = lambda: default_timer() - start
        yield lambda: elapser()
        end = default_timer()
        elapser = lambda: end-start

    alpha, min_alpha, passes = (0.025, 0.001, 20)
    alpha_delta = (alpha - min_alpha) / passes 
    print("Training {}...".format(model))
    for epoch in range(passes):
        duration = 'na'
        model.alpha, model.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            model.train(Xa, total_examples=len(Xa), epochs=1)
            duration = '%.1f' % elapsed()        
            print('Completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta

    print("END: %s" % str(datetime.datetime.now()))
    end = time.time()
    print("Time elapsed: {0}".format(end-st))
    
    model.save(str(model).replace('/',''))
    
def train_tf_idf_model(Xa, min_df, max_df, n_components):
    pipe = Pipeline([
               ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
               ('tfidf', TfidfVectorizer(min_df=min_df, max_df=max_df, sublinear_tf=True)), 
               ('nmf', NMF(n_components=n_components))
           ])
    pipe.fit(Xa)
    file_name = 'tfidf({:02.1f},{:02.1f},{:03d})'.format(min_df, max_df, n_components) + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(pipe, f)

cores = -1
def main():
    dataset = utils_v2.read_dataset_dictionary(label_horizon=1, subset='full', momentum_only=False, doc2vec=False)
    Xa = np.empty([len(dataset['X'])], dtype=object)
    for i, x in enumerate(dataset['X']):
        tags = [i]
        Xa[i] = gensim.models.doc2vec.TaggedDocument(x['corpus'], tags)

    cores = multiprocessing.cpu_count()
    print("{} cores".format(cores))
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
    
    # Le and Mikolov 2014
    # "DM is better than DBOW, concat is better than average, window size can
    # be between 5-12 (window=3 to window=6). Best performance from combining
    # DM + DBOW"
    # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
    # "negative sampling: 5-20 for small datasets, 3-5 for large datasets"
    model_types = ['dm_mean', 'dm_concat', 'dbow']
    sizes = [100, 300, 600]
    windows = [3, 4, 5, 6]
    
#    for model_type in model_types:
#        for size in sizes:
#            for window in windows:
#                train_model(Xa, model_type, size, window)
    max_dfs = [0.8, 0.9, 1.0]
    min_dfs = [0.2, 0.3, 0.4]
    nmf_n_components = [50, 100, 300, 600]
    for max_df in max_dfs:
        for min_df in min_dfs:
            for n_components in nmf_n_components:
                train_tf_idf_model(dataset['X'], min_df, max_df, n_components)

if __name__ == '__main__':
    main()
