import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from xgboost import XGBClassifier
import custom_transformers

estimators_prices_xgb = [
    # Price history features
    ('price_history', Pipeline([
        ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
    ])),

    # Use a SVC classifier on the combined features
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',
                n_jobs=1,
                nthread=1,)),
]

"""https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
http://xgboost.readthedocs.io/en/latest/parameter.html
xgboost parameters
"""
param_grid_prices_xgb = dict(
    clf__max_depth=np.array([3,4,5,6,7,8,9]),
    clf__min_child_weight=np.array([3,4,5,6,7,8,9]),
    clf__subsample=np.array([0.5,0.6,0.7,0.8,0.9]),
    clf__colsample_bytree=np.array([0.5,0.6,0.7,0.8,0.9]),
)

"""http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
heterogeneous data pipeline
"""

"""Based off sample statistics from 2017/QTR1,
NMF(TFIDF, n_components=200): Batch of 1000
    Mean: 0.0029611563904979304
    STD: 0.029393819850990936
Prices (outliers removed): Batch of 1000
    Mean: -6.367020755381153e-05
    STD: 0.029223920127052702
No rescaling is necessary...
"""

estimators_tfidf_nmf_prices_xgb = [
    # Use feature union to combine linguistic features and price history features
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling linguistic features from Form 10-Ks
            ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
                # https://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html
                ('tfidf', TfidfVectorizer()), 
                ('nmf', NMF()),
            ])),
            # Price history features
            ('price_history', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
            ])),
            
        ],
        # https://stackoverflow.com/questions/29504252/whats-the-use-of-transformer-weights-in-scikit-learn-pipeline
        # We know nothing a priori about the weighting of features, so set transformer_weights = 1.0
        # TODO: Actually, we can use these transformer weights to normalize different features!
        #
        transformer_weights={
            'linguistic': 1.0,
            'price_history': 1.0
        },
    )),

    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',
                n_jobs=1,
                nthread=1,)),
]

estimators_tfidf_nmf_prices_xgb = [
    # Use feature union to combine linguistic features and price history features
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling linguistic features from Form 10-Ks
            ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
                # https://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html
                #
                ('tfidf', TfidfVectorizer()), 
                ('nmf', NMF()),
            ])),
            # Price history features
            ('price_history', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
            ])),
            
        ],
        # https://stackoverflow.com/questions/29504252/whats-the-use-of-transformer-weights-in-scikit-learn-pipeline
        # We know nothing a priori about the weighting of features, so set transformer_weights = 1.0
        # TODO: Actually, we can use these transformer weights to normalize different features!
        #
        transformer_weights={
            'linguistic': 1.0,
            'price_history': 1.0
        },
    )),

    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',
                n_jobs=1,
                nthread=1,)),
]


"""
param_grid_tfidf_nmf_prices_xgb = dict(
    clf__max_depth=np.array([6]),
    clf__min_child_weight=np.array([6]),
    clf__subsample=np.array([0.7]),
    clf__colsample_bytree=np.array([0.7]),
    union__linguistic__tfidf__max_df=np.array([0.7]),
    union__linguistic__tfidf__min_df=np.array([0.2]),
    union__linguistic__tfidf__sublinear_tf=[True],
    union__linguistic__nmf__n_components=np.array([200]),
)
"""

param_grid_tfidf_nmf_prices_xgb = dict(
    clf__max_depth=np.array([3,4,5,6,7,8,9]),
    clf__min_child_weight=np.array([3,4,5,6,7,8,9]),
    clf__subsample=np.array([0.5,0.6,0.7,0.8,0.9]),
    clf__colsample_bytree=np.array([0.5,0.6,0.7,0.8,0.9]),
    union__linguistic__tfidf__max_df=np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
    union__linguistic__tfidf__min_df=np.array([0.1, 0.2, 0.3]),
    union__linguistic__tfidf__sublinear_tf=[True],
    union__linguistic__nmf__n_components=np.array([100, 150, 200, 250, 300, 350, 400]),
)

estimators_doc2vec_prices_xgb = [
    # Use feature union to combine linguistic features and price history features
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling linguistic features from Form 10-Ks
            ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='doc2vec')),
                # https://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html
                #
                #('doc2vec', custom_transformers.Doc2VecVectorizer()),
                #('debug', custom_transformers.DebugTransformer())
            ])),
            # Price history features
            ('price_history', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
            ])),
            
        ],
        # Actually, we can use these transformer weights to normalize different features!
        # dmc STD: .00288
        # momentum STD: .028 => linguistic feature scaling = 10.0
        transformer_weights={
            'linguistic': 10.0,
            'price_history': 1.0
        },
    )),

    # Use a SVC classifier on the combined features
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',)),
]

"""http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
tfidf parameters
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
nmf parameters
"""
"""
param_grid_doc2vec_prices_xgb = dict(
    clf__max_depth=np.array([3,4,5,6,7,8,9]),
    clf__min_child_weight=np.array([3,4,5,6,7,8,9]),
    clf__subsample=np.array([0.5,0.6,0.7,0.8,0.9]),
    clf__colsample_bytree=np.array([0.5,0.6,0.7,0.8,0.9]),
)
"""
param_grid_doc2vec_prices_xgb = dict(
    clf__max_depth=np.array([7]),
    clf__min_child_weight=np.array([7]),
    clf__subsample=np.array([0.7]),
    clf__colsample_bytree=np.array([0.7]),
)

