import pickle
import os
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
from itertools import tee
def split_gen(gen):
    gen_a, gen_b = tee(gen, 2)
    return (a for a, b in gen_a), (b for a, b in gen_b)
def split_gen_6(gen):
    gen_a, gen_b, gen_c, gen_d, gen_e, gen_f = tee(gen, 6)
    return (a for a, b, c, d, e, f in gen_a), (b for a, b, c, d, e, f in gen_b), (c for a, b, c, d, e, f in gen_c), (d for a, b, c, d, e, f in gen_d), (e for a, b, c, d, e, f in gen_e), (f for a, b, c, d, e, f in gen_f)
def main():
    print('process starting...')
    gen = utils_v2.load_data(data_dir, split='all') 
    corpus, labels = split_gen(gen) 
#    for item in corpus:
#        print(item[:10])
#   for label_type in labels:
#        for label in label_type:
#            print(label)
    baseline, alpha1, alpha2, alpha3, alpha, alpha5 = split_gen_6(labels)
    pipe = Pipeline([('tfidf', TfidfVectorizer), ('nmf', NMF), ('clf', SVC)])
    """https://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html
    """
    param_grid = dict(vectorizer=[TfidfVectorizer(sublinear_tf=True)],
                      reduce_dim=[NMF(n_components=50), NMF(n_components=100), NMF(n_components=200)],
                      clf=[SVC(C=0.1), SVC(C=10), SVC(C=100),
                           LogisticRegression(C=0.1), LogisticRegression(C=10), LogisticRegression(C=100), 
                           MultinomialNB(), 
                           XGBClassifier()])
    """https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
    """
    ts_cv = TimeSeriesSplit(n_splits=2).split(corpus)
    """https://stackoverflow.com/questions/46732748/how-do-i-use-a-timeseriessplit-with-a-gridsearchcv-object-to-tune-a-model-in-sci
    """
    grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=ts_cv)
    grid_search.fit(corpus, alpha1) 

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
