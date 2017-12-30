from gensim.models import Doc2Vec
import numpy as np
import os.path
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
        
print("Starting...")
sys.stdout.flush()
dirname = 'data_by_returns'

model = Doc2Vec.load('saved_doc2vec_models/Doc2Vec(dmm,d100,n5,w10,mc2,s0.001,t8)')
docs = read_corpus(os.path.join(dirname, 'alldata-id.txt'))
labels = read_corpus_labels(os.path.join(dirname, 'alldata-id.txt'))

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import pickle
clfs = [XGBClassifier()]
param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}
for clf in clfs:
    print(clf) 
    gsearch1 = GridSearchCV(estimator = XGBClassifier(
        learning_rate = 0.1,
        n_estimators = 140,
        silent = True,
        objective = 'binary:logistic',
        nthread = 4),
                           param_grid = param_test1,
                           scoring = 'roc_auc',
                           n_jobs=4,
                           iid=True,
                           cv=5,
                           refit = True
                           )
    gsearch1.fit(np.array([model.infer_vector(doc) for doc in docs]), np.array([int(label) for label in labels]))
    picklepath = 'xgbgridsearch.pkl'
    with open(picklepath, 'wb') as f:
        pickle.dump(gsearch1, f, protocol=pickle.HIGHEST_PROTOCOL)
    #print(gsearch1.grid_scores_)
    #print(gsearch1.best_params_)
    #print(gsearch1.best_score_)           
