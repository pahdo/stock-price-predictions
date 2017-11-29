import pickle
from sklearn.model_selection import GridSearchCV
picklepath = 'xgbgridsearch.pkl'
with open(picklepath, 'rb') as f:
    gsearch1 = pickle.loads(f.read())
    print(gsearch1.grid_scores_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)
from gensim.models import Doc2Vec
model = Doc2Vec.load('saved_doc2vec_models/Doc2Vec(dmm,d100,n5,w10,mc2,s0.001,t8)')
import sys
def read_corpus(f):
    with open(f) as file:
        for line_no, line in enumerate(file):
            if ((line_no)%200==0):
                print("read line {}".format(line_no))
                sys.stdout.flush()
            line = line.split()
            if(len(line)==0):
                continue
            line = line[2:]
            line = ' '.join(line)
            yield line
def read_corpus_labels(f):
    with open(f) as file:
        for line_no, line in enumerate(file):
            if ((line_no)%200==0):
                print("read sentiment {}".format(line_no))
                sys.stdout.flush()
            line = line.split()
            if(len(line)==0):
                continue
            line = line[1] # SENTIMENT!
            yield line
dirname = 'data_by_returns_small'
import os.path
docs = read_corpus(os.path.join(dirname, 'alldata-id.txt'))
labels = read_corpus_labels(os.path.join(dirname, 'alldata-id.txt'))
from sklearn.metrics import accuracy_score
import numpy as np
docs = [np.array(model.infer_vector(doc)) for doc in docs]
labels = [int(label) for label in labels]
score = gsearch1.score(np.array(docs), labels)
predictions = gsearch1.predict(np.array(docs))
print("accuracy {}".format(accuracy_score(labels, predictions)))
