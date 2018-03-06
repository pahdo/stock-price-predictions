import pickle
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

def read_dataset(label_horizon):
    """
    args : 
        label_horizon : 1, 2, 3, 4, 5 to decide between alpha1, alpha2, etc.
    """
    data_index_path = 'data/' + my_config.dataset_dir + '/train.txt'
    # TODO: fix my data_index_path
    # with open(data_index_path, 'r') as data_index:
    with open('data/' + my_config.dataset_dir + '/train_qtr1.txt', 'r') as data_index:
        for path_prices_labels in data_index.readlines():
            path, prices, labels = path_prices_labels.split(';')
            prices = prices.split(',')

            """for detecting outliers
            """
            for p in prices:
                if float(p) > 1.0 or float(p) < -1.0:
                    print("string price: {}".format(p))

            labels = labels.split(',')
            with open(path, 'r') as t:
                yield t.read(), [float(p) for p in prices], utils_v2.bin_alpha(float(labels[label_horizon]))

def read_dataset_dictionary(label_horizon):
    dataset_gen = read_dataset(label_horizon)
    text, prices, labels = utils_v2.split_gen_3(dataset_gen)

    X = []
    for t in text:
        price_hist = np.array(next(prices))
        X.append({'corpus': t, 'price_history': price_hist})

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
pickle_path = 'tfidf_best_estimator.pkl'
best_estimator = joblib.load(pickle_path)
dataset = read_dataset_dictionary(label_horizon=1)
print(best_estimator.score(dataset['X'], dataset['labels']) )