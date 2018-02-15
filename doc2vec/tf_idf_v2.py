import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
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
import custom_transformers
import my_config
import utils_v2

def main():
    gen = utils_v2.load_data(my_config.data_dir, 'all', my_config.train_quarters, my_config.test_quarters) 
    features, labels = utils_v2.split_gen(gen)
    corpus, price_history = utils_v2.split_gen(features)
    alpha1, alpha2, alpha3, alpha, alpha5 = utils_v2.split_gen_5(labels)

    corpus = [item for item in corpus]
    price_history = [item for item in price_history]
    data = {}
    data['corpus'] = corpus
    data['price_history'] = price_history
    alpha1 = [item for item in alpha1]
    bins1 = [utils_v2.bin_alpha(item) for item in alpha1]
    print("DATASET SIZE: {}".format(len(corpus)))

    estimators = [
        # Use feature union to combine linguistic features and price history features
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for pulling linguistic features from Form 10-Ks
                ('linguistic', Pipeline([
                    ('selector', custom_transformers.ItemSelector(key='corpus')),
                    ('tfidf', TfidfVectorizer(sublinear_tf=True)), 
                    ('nmf', NMF(n_components=25)),
                ])),
                # Price history features
                ('price_history', Pipeline([
                    ('selector', custom_transformers.ItemSelector(key='price_history')),
                ])),
                
            ],
            transformer_weights={
                'linguistic': 1.0,
                'price_history': 1.0
            },
        )),

        # Use a SVC classifier on the combined features
        ('clf', SVC(kernel='rbf')),
    ]
    Cs = np.logspace(-6, -1, 10)
    run_experiment(estimators, Cs, corpus, bins1)

def run_experiment(estimators, param_dict, features, labels):
    print('experiment starting with estimators={} param_dict={} features={} labels = {}...'.format(estimators, param_dict, '...', '...'))
    # TODO: Feature union - create transformer obj. for baseline
    pipe = Pipeline(memory=my_config.cache_dir, steps=estimators)
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
    """https://stackoverflow.com/questions/29504252/whats-the-use-of-transformer-weights-in-scikit-learn-pipeline
    We know nothing a priori about the weighting of features, so set transformer_weights = 1.0
    """

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
