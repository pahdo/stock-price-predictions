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

def main():
    st = time.time() 
    dataset = utils_v2.read_dataset_dictionary(label_horizon=1, subset='full', momentum_only=False, doc2vec=False)
    Xa = np.empty([len(dataset['X'])], dtype=object)
    for i, x in enumerate(dataset['X']):
        tags = [i]
        Xa[i] = gensim.models.doc2vec.TaggedDocument(x['corpus'], tags)

    cores = multiprocessing.cpu_count()
    print("{} cores".format(cores))
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
    
    simple_models = [
        # DM is better than DBOW
        # Average is better than concat
        # Window size can be 5 - 12
        Doc2Vec(dm=1, dm_mean=1, size=100, window=8, max_vocab_size=100000, negative=5, hs=0, min_count=2, workers=cores),
    ]
    
    print("Building a vocabulary...")
    start_tm = time.time()
    simple_models[0].build_vocab(Xa)
    end_tm = time.time()
    print("Vocabulary built for the first time in {0}".format(end_tm-start_tm))
    
    for model in simple_models:
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
    for epoch in range(passes):
        for model in simple_models: 
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

    for model in simple_models:
        model.save(str(model).replace('/',''))

if __name__ == '__main__':
    main()
