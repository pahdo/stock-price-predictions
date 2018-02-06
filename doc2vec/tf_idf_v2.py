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

#train_quarters = ['2005/QTR1', '2005/QTR2', '2005/QTR3', '2005/QTR4']
train_quarters = [
    '2005/QTR1', '2005/QTR2', '2005/QTR3', '2005/QTR4',
    '2004/QTR1', '2004/QTR2', '2004/QTR3', '2004/QTR4',
    '2003/QTR1', '2003/QTR2', '2003/QTR3', '2003/QTR4',
    '2002/QTR1', '2002/QTR2', '2002/QTR3', '2002/QTR4',
    '2001/QTR1', '2001/QTR2', '2001/QTR3', '2001/QTR4',
    '2000/QTR1', '2000/QTR2', '2000/QTR3', '2000/QTR4',
    '1999/QTR1', '1999/QTR2', '1999/QTR3', '1999/QTR4',
    '1998/QTR1', '1998/QTR2', '1998/QTR3', '1998/QTR4',
    '1997/QTR1', '1997/QTR2', '1997/QTR3', '1997/QTR4',
    '1996/QTR1', '1996/QTR2', '1996/QTR3', '1996/QTR4',
    '1995/QTR1', '1995/QTR2', '1995/QTR3', '1995/QTR4',
    '1994/QTR1', '1994/QTR2', '1994/QTR3', '1994/QTR4',
    '2009/QTR1', '2009/QTR2', '2009/QTR3', '2009/QTR4',
    '2008/QTR1', '2008/QTR2', '2008/QTR3', '2008/QTR4',
    '2007/QTR1', '2007/QTR2', '2007/QTR3', '2007/QTR4',
    '2006/QTR1', '2006/QTR2', '2006/QTR3', '2006/QTR4']
test_quarters = ['2013/QTR2']
"""
test_quarters = [
    '2013/QTR2', '2013/QTR3', '2013/QTR4',
    '2012/QTR1', '2012/QTR2', '2012/QTR3', '2012/QTR4',
    '2011/QTR1', '2011/QTR2', '2011/QTR3', '2011/QTR4',
    '2010/QTR1', '2010/QTR2', '2010/QTR3', '2010/QTR4']
"""

data_dir = '10-X_C_clean'
output_dir = '10-X_C_clean'

###########################

"""https://stackoverflow.com/questions/28030095/how-to-split-a-python-generator-of-tuples-into-2-separate-generators
"""
def split_gen(gen):
    gen_a, gen_b = tee(gen, 2)
    return (a for a, b in gen_a), (b for a, b in gen_b)

def split_gen_5(gen):
    gen_a, gen_b, gen_c, gen_d, gen_e = tee(gen, 5)
    return (a for a, b, c, d, e in gen_a), (b for a, b, c, d, e in gen_b), (c for a, b, c, d, e in gen_c), (d for a, b, c, d, e in gen_d), (e for a, b, c, d, e in gen_e)

def bin_alpha(a):
    threshold = 0.01
    if a < -1 * threshold:
        return -1
    elif a > threshold:
        return 1
    else:
        return 0

def main():
    gen = utils_v2.load_data(data_dir, 'all', train_quarters, test_quarters) 
    features, labels = split_gen(gen)
    corpus, price_history = split_gen(features)
    alpha1, alpha2, alpha3, alpha, alpha5 = split_gen_5(labels)
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
