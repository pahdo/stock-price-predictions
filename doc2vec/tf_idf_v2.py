import pickle
import os
import time
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

def main():
    dataset = read_dataset_dictionary(label_horizon=1)

    # param_grid = dict(vectorizer=[('tfidf', TfidfVectorizer(sublinear_tf=True)],
    #                   reduce_dim=[NMF(n_components=50), NMF(n_components=100), NMF(n_components=200)],
    #                   clf=[SVC(C=0.1), SVC(C=10), SVC(C=100),
    #                        LogisticRegression(C=0.1), LogisticRegression(C=10), LogisticRegression(C=100), 
    #                        MultinomialNB(), 
    #                        XGBClassifier()])
    #Cs = np.logspace(-6, -1, 10)
    param_grid = my_estimators.param_grid_tfidf_nmf_prices_xgb
    estimators = my_estimators.estimators_tfidf_nmf_prices_xgb
    # param_grid = my_estimators.param_grid_prices_xgb
    # estimators = my_estimators.estimators_prices_xgb
    run_experiment(estimators, param_grid, dataset)

def run_experiment(estimators, param_dict, dataset):
    print('experiment starting with estimators={} param_dict={}'.format(estimators, param_dict))
    start = time.time()
    pipe = Pipeline(memory=my_config.cache_dir, steps=estimators)

    """https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
    """
    """https://stackoverflow.com/questions/46732748/how-do-i-use-a-timeseriessplit-with-a-gridsearchcv-object-to-tune-a-model-in-sci
    """
    ts_cv = TimeSeriesSplit(n_splits=2).split(dataset['X'])

    #grid_search = GridSearchCV(pipe, param_grid=dict(clf__C=param_dict), cv=ts_cv)
    grid_search = GridSearchCV(pipe, param_grid=param_dict, cv=ts_cv, n_jobs=8)
    print(len(dataset['X']))
    print(len(dataset['labels']))
    grid_search.fit(dataset['X'], dataset['labels']) 
    end = time.time()
    print("Total running time: {}".format(end-start))
    print(grid_search.grid_scores_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    from sklearn.externals import joblib
    joblib.dump(grid_search.best_estimator_, 'tfidf_best_estimator.pkl', compress=1)

if __name__ == "__main__":
    main()
