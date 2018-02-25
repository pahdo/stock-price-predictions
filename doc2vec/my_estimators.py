import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from xgboost import XGBClassifier
import custom_transformers

"""http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
heterogeneous data pipeline
"""
estimators_tfidf_nmf_prices_svc = [
    # Use feature union to combine linguistic features and price history features
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling linguistic features from Form 10-Ks
            ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
                # https://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html
                #
                ('tfidf', TfidfVectorizer(sublinear_tf=True)), 
                ('nmf', NMF(n_components=25)),
            ])),
            # Price history features
            ('price_history', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
            ])),
            
        ],
        # https://stackoverflow.com/questions/29504252/whats-the-use-of-transformer-weights-in-scikit-learn-pipeline
        # We know nothing a priori about the weighting of features, so set transformer_weights = 1.0
        #
        transformer_weights={
            'linguistic': 1.0,
            'price_history': 1.0
        },
    )),

    # Use a SVC classifier on the combined features
    ('clf', SVC(kernel='rbf')),
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
                ('tfidf', TfidfVectorizer(sublinear_tf=False)), 
                ('nmf', NMF(n_components=True)),
            ])),
            # Price history features
            ('price_history', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
            ])),
            
        ],
        # https://stackoverflow.com/questions/29504252/whats-the-use-of-transformer-weights-in-scikit-learn-pipeline
        # We know nothing a priori about the weighting of features, so set transformer_weights = 1.0
        #
        transformer_weights={
            'linguistic': 1.0,
            'price_history': 1.0
        },
    )),

    # Use a SVC classifier on the combined features
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 120,
                silent = True,
                objective = 'multi:softmax',
                nthread = 4)),
]

"""http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
tfidf parameters
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
nmf parameters
"""
param_grid_prices_xgb = dict(clf__max_depth=np.arange(2,6,1),
    clf__learning_rate=np.logspace(-6,-1,10),
    clf__n_estimators=np.arange(40, 320, 20),
    clf__lambda=np.logspace(-4,4,10),
    tfidf__max_df=np.arange(.3, 1., .1),
    tfidf__min_df=np.arange(0., .2, .05),
    sublinear_tf=[True, False],
    nmf__n_components=np.arange(50,500,50),
)

estimators_prices_svc = [
    # Price history features
    ('price_history', Pipeline([
        ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
    ])),

    # Use a SVC classifier on the combined features
    ('clf', SVC(kernel='rbf')),
]

estimators_prices_xgb = [
    # Price history features
    ('price_history', Pipeline([
        ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
    ])),

    # Use a SVC classifier on the combined features
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 120,
                silent = True,
                objective = 'multi:softmax',
                nthread = 4)),
]

"""https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
http://xgboost.readthedocs.io/en/latest/parameter.html
xgboost parameters
"""
param_grid_prices_xgb = dict(clf__max_depth=np.arange(2,6,1),
    clf__learning_rate=np.logspace(-6,-1,10),
    clf__n_estimators=np.arange(40, 320, 20),
    clf__lambda=np.logspace(-4,4,10),
)

    # param_grid = dict(vectorizer=[('tfidf', TfidfVectorizer(sublinear_tf=True)],
    #                   reduce_dim=[NMF(n_components=50), NMF(n_components=100), NMF(n_components=200)],
    #                   clf=[SVC(C=0.1), SVC(C=10), SVC(C=100),
    #                        LogisticRegression(C=0.1), LogisticRegression(C=10), LogisticRegression(C=100), 
    #                        MultinomialNB(), 
    #                        XGBClassifier()])