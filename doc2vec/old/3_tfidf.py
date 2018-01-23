import sys

# Gives an iterable to the documents in the corpus
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
# Gives an iterable to the sentiments (1 or 0) in the corpus
def read_corpus_labels(f):
    with open(f) as file:
        for line_no, line in enumerate(file):
            if ((line_no)%25000==0):
                print("read sentiment {}".format(line_no))
                sys.stdout.flush()
            line = line.split()
            if(len(line)==0):
                continue
            line = line[1] # SENTIMENT!
            yield line
        
print("Starting...")
sys.stdout.flush()
dirname = 'data_by_returns_small'

from sklearn.feature_extraction.text import TfidfVectorizer
print("Reading through corpus and building word embeddings...")
sys.stdout.flush()
# Note documents and labels are read simultaneously from the same file, so they are in the correct order
import os.path
corpus = read_corpus(os.path.join(dirname, 'alldata-id.txt'))
labels = read_corpus_labels(os.path.join(dirname, 'alldata-id.txt'))
vectorizer = TfidfVectorizer(max_features=300, max_df=0.5, min_df=0.1, sublinear_tf=True)
embedded = vectorizer.fit_transform(corpus)

import pickle
pickle.dump(vectorizer, open("tfidf_vectorizer.pickle", "wb"))
pickle.dump(embedded, open("tfidf_embedded.pickle", "wb"))

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

clfs = [XGBClassifier()]
param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}
import numpy as np
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
    gsearch1.fit(embedded, np.array([int(label) for label in labels]))
    picklepath = 'xgbgridsearch_tfidf.pkl'
    with open(picklepath, 'wb') as f:
        pickle.dump(gsearch1, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(gsearch1.grid_scores_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)           
