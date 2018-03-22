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

def read_dataset(label_horizon, subset='full', doc2vec=False):
    """
    args : 
        label_horizon : 1, 2, 3, 4, 5 to decide between alpha1, alpha2, etc.
    """
    if subset == 'full' and not doc2vec:
        data_index_path = 'data/' + my_config.dataset_dir + '/all-train.txt'
    elif subset == 'full' and doc2vec:
        data_index_path = 'data/' + my_config.dataset_dir + '/all-doc2vec_test.txt'
    elif subset == '10000':
        data_index_path = 'data/' + my_config.dataset_dir + '/all-train_10000.txt'
    with open(data_index_path, 'r') as data_index:
        for path_prices_labels in data_index.readlines():
            if doc2vec:
                v, prices, labels = path_prices_labels.split('\t')
            else:
                path, prices, labels = path_prices_labels.split(';')
            prices = prices.split(',')

            """for detecting outliers
            """
            for p in prices:
                if float(p) > 1.0 or float(p) < -1.0:
                    print("string price: {}".format(p))

            labels = labels.split(',')
            if doc2vec:
                yield pickle.loads(codecs.decode(v.encode(), 'base64')), [float(p) for p in prices], utils_v2.bin_alpha(float(labels[label_horizon]))
            else:
                with open(path, 'r') as t:
                    yield t.read(), [float(p) for p in prices], utils_v2.bin_alpha(float(labels[label_horizon]))

def read_dataset_dictionary(label_horizon, subset='full', momentum_only=False, doc2vec=False):
    dataset_gen = read_dataset(label_horizon, subset, doc2vec)
    text, prices, labels = utils_v2.split_gen_3(dataset_gen)

    # TODO: For momentum_only, speed up by not opening files in dataset at all
    X = [] 
    for t in text:
        price_hist = np.array(next(prices))
        if momentum_only:
            X.append({'price_history': price_hist})
        else:
            X.append({'corpus': t, 'price_history': price_hist})
    X = np.array(X, dtype=object)

    dataset_size = len(X)
    print("dataset_size = {}".format(dataset_size))
    dataset = {}

    dataset['X'] = X

    """https://stackoverflow.com/questions/31995175/scikit-learn-cross-val-score-too-many-indices-for-array
    In sklearn cross-validation, labels must be (N,), not (N,1)
    """
    dataset['labels'] = np.array(list(labels))
    print("labels len = {}".format(len(dataset['labels'])))
    
    return dataset

from sklearn.externals import joblib
#pickle_path = 'momentum_full_1_best_estimator.pkl'
#pickle_path = 'tfidf_full_1_best_estimator.pkl'
pickle_path = 'doc2vec_full_1_best_estimator_no_gridsearch.pkl'
best_estimator = joblib.load(pickle_path)
dataset = read_dataset_dictionary(label_horizon=1, subset='full', momentum_only=False, doc2vec=True)
print(best_estimator.score(dataset['X'], dataset['labels']))
