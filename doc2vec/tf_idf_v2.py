import pickle
import os
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
import my_config
import my_estimators
import utils_v2
import my_diagnostics

def read_dataset(label_horizon, subset='full'):
    """
    args : 
        label_horizon : 1, 2, 3, 4, 5 to decide between alpha1, alpha2, etc.
    """
    if subset == 'full':
        data_index_path = 'data/' + my_config.dataset_dir + '/all-train_full.txt'
    elif subset == '10000':
        data_index_path = 'data/' + my_config.dataset_dir + '/all-train.txt'
    with open(data_index_path, 'r') as data_index:
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

def read_dataset_dictionary(label_horizon, subset='full'):
    dataset_gen = read_dataset(label_horizon, subset)
    text, prices, labels = utils_v2.split_gen_3(dataset_gen)

    X = [] 
    for t in text:
        price_hist = np.array(next(prices))
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

def get_estimators(key):
    param_grid = None
    estimators = None
    if key == 'doc2vec':
        """https://github.com/RaRe-Technologies/gensim/issues/1952
        Doc2Vec on gensim 3.4.0 (latest version) FAILS due to above error. Unresolved.
        Solution: downgrade to 3.2.0
        """
        param_grid = my_estimators.param_grid_doc2vec_prices_xgb
        estimators = my_estimators.estimators_doc2vec_prices_xgb
    elif key == 'tfidf':
        param_grid = my_estimators.param_grid_tfidf_nmf_prices_xgb
        estimators = my_estimators.estimators_tfidf_nmf_prices_xgb
    elif key == 'momentum':
        param_grid = my_estimators.param_grid_prices_xgb
        estimators = my_estimators.estimators_prices_xgb
    return estimators, param_grid

def main():
    label_horizon=1
    subset='full'
    key='momentum'
    dataset = read_dataset_dictionary(label_horizon=label_horizon, subset=subset)
    estimators, param_grid = get_estimators(key)
    pickle_path = key + '_' + subset + '_' + str(label_horizon) + '_best_estimator.pkl'
    
    run_experiment(estimators, param_grid, pickle_path, dataset)

def run_experiment(estimators, param_dict, pickle_path, dataset):
    print('experiment starting with estimators={} param_dict={}'.format(estimators, param_dict))
    start = time.time()
    pipe = Pipeline(memory=my_config.cache_dir, steps=estimators)

    """https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
    https://stackoverflow.com/questions/46732748/how-do-i-use-a-timeseriessplit-with-a-gridsearchcv-object-to-tune-a-model-in-sci
    """
    ts_cv = TimeSeriesSplit(n_splits=10).split(dataset['X'])  
    
    """GridSearch
    http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    If n_jobs was set to a value higher than one, the data is copied for each point in the grid (and not n_jobs times). This is done for efficiency reasons if individual jobs take very little time, but may raise errors if the dataset is large and not enough memory is available. A workaround in this case is to set pre_dispatch. Then, the memory is copied only pre_dispatch many times. A reasonable value for pre_dispatch is 2 * n_jobs.
    size of dataset_clean = 2.2GB
    memory of machine = 64GB
    """
    grid_search = RandomizedSearchCV(pipe, param_distributions=param_dict, cv=ts_cv, n_jobs=12, pre_dispatch='n_jobs')
#    grid_search = GridSearchCV(pipe, param_grid=param_dict, cv=ts_cv, n_jobs=24, pre_dispatch='n_jobs')
    print(len(dataset['X']))
    print(len(dataset['labels']))

    from sklearn.externals import joblib
    joblib.dump(dataset['X'], 'dataset_dump.pkl')
    dataset['X'] = joblib.load('dataset_dump.pkl', mmap_mode='c')
    
    grid_search.fit(dataset['X'], dataset['labels']) 

    end = time.time()
    print("Total running time: {}".format(end-start))
    print(grid_search.cv_results_)
    print(grid_search.grid_scores_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    from sklearn.externals import joblib
    joblib.dump(grid_search.best_estimator_, pickle_path, compress=1)

if __name__ == "__main__":
    main()
