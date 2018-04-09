import os
import time
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
import pickle
import codecs
import my_config
import my_estimators
import utils_v2
import my_diagnostics

def get_estimators(key):
    param_grid = None
    estimators = None
    if key == 'doc2vec':
        param_grid = my_estimators.param_grid_doc2vec_prices_xgb
        estimators = my_estimators.estimators_doc2vec_prices_xgb
        momentum_only = False
        doctag_only = False
        doc2vec = True
    elif key == 'dm_dbow_train':
        param_grid = my_estimators.param_grid_dm_dbow_xgb
        estimators = my_estimators.estimators_dm_dbow_xgb
        momentum_only = False
        doctag_only = True
        doc2vec = False
    elif key == 'tf_idf':
        param_grid = my_estimators.param_grid_tfidf_nmf_prices_xgb
        estimators = my_estimators.estimators_tfidf_nmf_prices_xgb
        momentum_only = False
        doctag_only = False
        doc2vec = False
    elif key == 'momentum':
        param_grid = my_estimators.param_grid_prices_xgb
        estimators = my_estimators.estimators_prices_xgb
        momentum_only = True
        doctag_only = False
        doc2vec = False
    elif key == 'dm_dbow_tfidf_1':
        estimators = my_estimators.estimators_dm_dbow_tfidf_xgb_1
        param_grid = my_estimators.param_grid_dm_dbow_tfidf_xgb
        momentum_only = False
        doctag_only = False
        doc2vec = False
    elif key == 'dm_dbow_tfidf_2':
        estimators = my_estimators.estimators_dm_dbow_tfidf_xgb_2
        param_grid = my_estimators.param_grid_dm_dbow_tfidf_xgb
        momentum_only = False
        doctag_only = False
        doc2vec = False
    elif key == 'dm_dbow_tfidf_3':
        estimators = my_estimators.estimators_dm_dbow_tfidf_xgb_3
        param_grid = my_estimators.param_grid_dm_dbow_tfidf_xgb
        momentum_only = False
        doctag_only = False
        doc2vec = False
    elif key == 'dm_dbow_tfidf_4':
        estimators = my_estimators.estimators_dm_dbow_tfidf_xgb_4
        param_grid = my_estimators.param_grid_dm_dbow_tfidf_xgb
        momentum_only = False
        doctag_only = False
        doc2vec = False
    elif key == 'ensemble':
        estimators = my_estimators.ensemble_clf
        param_grid = my_estimators.ensemble_param_grid
        momentum_only = False
        doctag_only = False
        doc2vec = False
    else:
        print("ERROR: INVALID KEY")
    return estimators, param_grid, momentum_only, doc2vec, doctag_only

def main():
    my_diagnostics.tracemalloc.start()
    label_horizon=1
    subset='full'
    key = 'ensemble'
    estimators, param_grid, momentum_only, doc2vec, doctag_only = get_estimators(key)
    dataset = get_dataset(label_horizon, subset, momentum_only, doc2vec, doctag_only)
    pickle_path = key + '_' + subset + '_' + str(label_horizon) + '_best_estimator.pkl'
    
    run_experiment(estimators, param_grid, pickle_path, dataset)

"""large local variables to garbage collected when this function returns
before: 8GB of memory used before forking
after: 3.5GB of memory used before forking
"""
def get_dataset(label_horizon, subset, momentum_only, doc2vec, doctag_only):
    dataset = utils_v2.read_dataset_dictionary(label_horizon=label_horizon, subset=subset, momentum_only=momentum_only, doc2vec=doc2vec, doctag_only=doctag_only)
    joblib.dump(dataset['X'], 'dataset_dump.pkl')
    dataset['X'] = joblib.load('dataset_dump.pkl', mmap_mode='r')
    return dataset

def run_experiment(estimators, param_dict, pickle_path, dataset):
    dataset['X'] = dataset['X']
    dataset['labels'] = dataset['labels']
    
    print('experiment starting with estimators={} param_dict={}'.format(estimators, param_dict))
    start = time.time()
    pipe = Pipeline(memory=my_config.cache_dir, steps=estimators)

    """https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
    https://stackoverflow.com/questions/46732748/how-do-i-use-a-timeseriessplit-with-a-gridsearchcv-object-to-tune-a-model-in-sci
    """
    ts_cv = TimeSeriesSplit(n_splits=2).split(dataset['X'])  
    
    """GridSearch
    http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    If n_jobs was set to a value higher than one, the data is copied for each point in the grid (and not n_jobs times). This is done for efficiency reasons if individual jobs take very little time, but may raise errors if the dataset is large and not enough memory is available. A workaround in this case is to set pre_dispatch. Then, the memory is copied only pre_dispatch many times. A reasonable value for pre_dispatch is 2 * n_jobs.
    """
#    grid_search = RandomizedSearchCV(pipe, param_distributions=param_dict, cv=ts_cv, n_jobs=24, pre_dispatch='n_jobs+4')
#    grid_search = RandomizedSearchCV(pipe, param_distributions=param_dict, n_iter=3, cv=ts_cv, n_jobs=3, pre_dispatch='n_jobs')
    snapshot = my_diagnostics.tracemalloc.take_snapshot()
    my_diagnostics.display_top(snapshot)
    
    print(len(dataset['X']))
    print(len(dataset['labels']))
    grid_search = GridSearchCV(pipe, param_grid=param_dict, cv=ts_cv)
    
    grid_search.fit(dataset['X'], dataset['labels']) 

    end = time.time()
    print("Total running time: {}".format(end-start))
    print(grid_search.cv_results_)
    print(grid_search.grid_scores_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    with open(pickle_path, 'wb+') as p:
        pickle.dump(grid_search.best_estimator_, p)

if __name__ == '__main__':
    main()
