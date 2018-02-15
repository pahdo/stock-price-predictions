from itertools import tee
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import utils_v2

def main():
    corpus = [item for item in corpus]
    price_history = [item for item in price_history]
    alpha1 = [item for item in alpha1]
    bins1 = [bin_alpha(item) for item in alpha1]
    print("DATASET SIZE: {}".format(len(corpus)))

    estimators = [('clf', SVC())]
    Cs = np.logspace(-6, -1, 10)
    run_experiment(estimators, Cs, price_history, bins1)

    estimators = [('tfidf', TfidfVectorizer(sublinear_tf=True)), ('nmf', NMF(n_components=25)), ('clf', SVC())]
    Cs = np.logspace(-6, -1, 10)
    run_experiment(estimators, Cs, corpus, bins1)

def run_experiment(estimators, param_dict, features, labels):
    print('experiment starting with estimators={} param_dict={} features={} labels = {}...'.format(estimators, param_dict, '...', '...'))
    # TODO: Feature union - create transformer obj. for baseline
    pipe = Pipeline(estimators)
    """https://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html
    """
    # param_grid = dict(vectorizer=[('tfidf', TfidfVectorizer(sublinear_tf=True)],
    #                   reduce_dim=[NMF(n_components=50), NMF(n_components=100), NMF(n_components=200)],
    #                   clf=[SVC(C=0.1), SVC(C=10), SVC(C=100),
    #                        LogisticRegression(C=0.1), LogisticRegression(C=10), LogisticRegression(C=100), 
    #                        MultinomialNB(), 
    #                        XGBClassifier()])
    """https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
    """
    ts_cv = TimeSeriesSplit(n_splits=2).split(features)
    """https://stackoverflow.com/questions/46732748/how-do-i-use-a-timeseriessplit-with-a-gridsearchcv-object-to-tune-a-model-in-sci
    """
    #grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=ts_cv)

    grid_search = GridSearchCV(pipe, param_grid=dict(clf__C=param_dict), cv=ts_cv)
    grid_search.fit(features, labels) 
    print(grid_search.grid_scores_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

#    pickle_path = 'gridsearch_tfidf_v2.pkl'
#    with open(pickle_path, 'wb') as f:
#        pickle.dump(grid_search, f, protocol=pickle.HIGHEST_PROTOCOL)

    # from sklearn import metrics

    # clfs = []
    # param_test1 = {
    #     'max_depth': range(3, 10, 2),
    #     'min_child_weight': range(1, 6, 2)
    # }
    # import numpy as np
    # for clf in clfs:
    #     print(clf) 
    #     gsearch1 = GridSearchCV(estimator = XGBClassifier(
    #         learning_rate = 0.1,
    #         n_estimators = 140,
    #         silent = True,
    #         objective = 'binary:logistic',
    #         nthread = 4),
    #                         param_grid = param_test1,
    #                         scoring = 'roc_auc',
    #                         n_jobs=4,
    #                         iid=True,
    #                         cv=5,
    #                         refit = True
    #                         )
    #     gsearch1.fit(embedded, np.array([int(label) for label in labels]))
    #     picklepath = 'xgbgridsearch_tfidf.pkl'
    #     with open(picklepath, 'wb') as f:
    #         pickle.dump(gsearch1, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     print(gsearch1.grid_scores_)
    #     print(gsearch1.best_params_)
    #     print(gsearch1.best_score_)

if __name__ == "__main__":
    main()
