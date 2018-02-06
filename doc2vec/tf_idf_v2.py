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

###### CONFIGURATION ######

data_dir = '10-X_C_clean'
output_dir = '10-X_C_clean'

###########################

"""https://stackoverflow.com/questions/28030095/how-to-split-a-python-generator-of-tuples-into-2-separate-generators
"""
def split_gen(gen):
    gen_a, gen_b = tee(gen, 2)
    return (a for a, b in gen_a), (b for a, b in gen_b)

def split_gen_6(gen):
    gen_a, gen_b, gen_c, gen_d, gen_e, gen_f = tee(gen, 6)
    return (a for a, b, c, d, e, f in gen_a), (b for a, b, c, d, e, f in gen_b), (c for a, b, c, d, e, f in gen_c), (d for a, b, c, d, e, f in gen_d), (e for a, b, c, d, e, f in gen_e), (f for a, b, c, d, e, f in gen_f)

def bin_alpha(a):
    threshold = 0.01
    if a < -1 * threshold:
        return -1
    elif a > threshold:
        return 1
    else:
        return 0

def main():
    print('process starting...')
    gen = utils_v2.load_data(data_dir, split='all') 
    corpus, labels = split_gen(gen)
    corpus = [doc for doc in corpus]
    print("corpus len={}".format(len(corpus)))
#    for item in corpus:
#        print(item[:10])
#   for label_type in labels:
#        for label in label_type:
#            print(label)
    baseline, alpha1, alpha2, alpha3, alpha, alpha5 = split_gen_6(labels)
    alpha1 = [label for label in alpha1]
    labels1 = list(map(bin_alpha, alpha1))
    print(labels1)
    print("labels1 len={}".format(len(labels1)))
    baseline = list(baseline)
    for item in baseline:
        assert len(item) == 5
    corpus = list(corpus)
    print(baseline)
    print(len(baseline))
# TODO: Feature union - create transformer obj. for baseline
    estimators = [('tfidf', TfidfVectorizer(sublinear_tf=True)), ('nmf', NMF(n_components=25)), ('clf', SVC())]
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
    ts_cv = TimeSeriesSplit(n_splits=2).split(corpus)
    """https://stackoverflow.com/questions/46732748/how-do-i-use-a-timeseriessplit-with-a-gridsearchcv-object-to-tune-a-model-in-sci
    """
    #grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=ts_cv)
    estimators = [('clf', SVC())]
    pipe = Pipeline(estimators)
    Cs = np.logspace(-6, -1, 10)
    grid_search = GridSearchCV(pipe, param_grid=dict(clf__C=Cs), cv=ts_cv)
    grid_search.fit(baseline, labels1) 
    print(grid_search.grid_scores_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    Cs = np.logspace(-6, -1, 10)
    grid_search = GridSearchCV(pipe, param_grid=dict(clf__C=Cs), cv=ts_cv)
    grid_search.fit(corpus, labels1) 
    pickle_path = 'gridsearch_tfidf_v2.pkl'
    print(grid_search.grid_scores_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
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
