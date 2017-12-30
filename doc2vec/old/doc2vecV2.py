from multiprocessing import Pool
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

nlp = spacy.load('en')
dirname = 'data_by_returns_small'

num_docs = file_len(os.path.join(dirname, 'alldata-id.txt'))
print("num_docs={}".format(num_docs))

def read_corpus(f):
    with open(f) as file:
        for line_no, line in enumerate(file):
            if ((line_no)%25000==0):
                print("read line {}".format(line_no))
                sys.stdout.flush()
            line = line.split()
            if(len(line)==0):
                continue
            line = line[2:]
            line = ' '.join(line)
            yield line

cores = multiprocessing.cpu_count()
print("{} cores".format(cores))
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

simple_models = [
    # PV-DM w/ average - DM has better word vector representations than DBOW and using average vs. concat means a smaller model
    # It is valuable to cross-validate the window size, a reasonable reason is 5-12
    # We can ignore document tags and use infer vector for the following classification step
    Doc2Vec(dm=1, dm_mean=1, size=100, window=8, max_vocab_size=100000, negative=5, hs=0, min_count=2, workers=cores),
]

print("Building a vocabulary...")
start_tm = time.time()
simple_models[0].build_vocab(read_corpus(os.path.join(dirname, 'alldata-id.txt')))
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
    
for epoch in range(passes):
    for model in simple_models: 
        docs = read_corpus(os.path.join(dirname, 'alldata-id.txt'))
        duration = 'na'
        model.alpha, model.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            model.train(docs, total_examples=num_docs, epochs=1)
            duration = '%.1f' % elapsed()        
    print('Completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta

print("END: %s" % str(datetime.datetime.now()))
end = time.time()
print("Time elapsed: {0}".format(end-start))
    
for model in simple_models:
    model.save(str(model).replace('/',''))
