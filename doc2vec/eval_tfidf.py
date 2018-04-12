import pickle
import codecs
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
import my_config
import my_estimators
import utils_v2


from sklearn.externals import joblib
pickle_path = 'ensemble_full_1_best_estimator.pkl'
dataset = utils_v2.read_dataset_dictionary(label_horizon=1, subset='full', momentum_only=False, doc2vec=False, testing=True)
with open(pickle_path, 'rb') as p:   
    best_estimator = pickle.load(p)
    result = best_estimator.score(dataset['X'], dataset['labels'])
    print(pickle_path)
    print(result)
    with open('eval_results.txt', 'a+') as t:
        t.write(pickle_path)
        t.write(str(result))
