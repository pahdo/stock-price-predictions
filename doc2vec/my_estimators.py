import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from xgboost import XGBClassifier
import custom_transformers
from sklearn.ensemble import VotingClassifier

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
                nthread=4,)),
]

"""
param_grid_tfidf_nmf_prices_xgb = dict(
    clf__max_depth=np.array([4]),
    clf__min_child_weight=np.array([6]),
    clf__subsample=np.array([0.6]),
    clf__colsample_bytree=np.array([0.7]),
    union__linguistic__tfidf__max_df=np.array([0.9]),
    union__linguistic__tfidf__min_df=np.array([0.3]),
    union__linguistic__tfidf__sublinear_tf=[True],
    union__linguistic__nmf__n_components=np.array([100]),
)
"""

param_grid_tfidf_nmf_prices_xgb = dict(
    clf__max_depth=np.array([2,3,4,5,6,7,8,9]),
    clf__min_child_weight=np.array([3,4,5,6,7,8,9]),
    clf__subsample=np.array([0.4,0.5,0.6,0.7,0.8,0.9]),
    clf__colsample_bytree=np.array([0.4,0.5,0.6,0.7,0.8,0.9]),
    union__linguistic__tfidf__max_df=np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]),
    union__linguistic__tfidf__min_df=np.array([0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]),
    union__linguistic__tfidf__sublinear_tf=[True],
    union__linguistic__nmf__n_components=np.array([25, 50, 75, 100, 125, 150, 200, 300, 400, 500, 600]),
)

estimators_doc2vec_prices_xgb = [
    # Use feature union to combine linguistic features and price history features
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling linguistic features from Form 10-Ks
            ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='doc2vec')),
                # https://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html
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
            'linguistic': 1.0,
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
param_grid_doc2vec_prices_xgb = dict(
    clf__max_depth=np.array([3,4,5,6,7,8,9]),
    clf__min_child_weight=np.array([3,4,5,6,7,8,9]),
    clf__subsample=np.array([0.5,0.6,0.7,0.8,0.9]),
    clf__colsample_bytree=np.array([0.5,0.6,0.7,0.8,0.9]),
)

estimators_dm_dbow_xgb = [
    ('union', FeatureUnion(
        transformer_list=[
            ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
                ('doc2vec', custom_transformers.DmDbowTrainVectorizer(
                                    dm_path='saved_doc2vec_models2/Doc2Vec(dbow,d300,n5,hs,mc2,s0.001)',
                                    dbow_path='saved_doc2vec_models2/Doc2Vec(dmc,d300,n5,hs,w5,mc2,s0.001)'
                                )),
            ])),
            ('price_history', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
            ])),        
        ],
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
                nthread=1,)),
]

param_grid_dm_dbow_xgb = dict(
    clf__max_depth=np.array([7]),
    clf__min_child_weight=np.array([7]),
    clf__subsample=np.array([0.7]),
    clf__colsample_bytree=np.array([0.7]),
)

estimators_dm_dbow_tfidf_xgb_1 = [
    ('union', FeatureUnion(
        transformer_list=[
            ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
                ('doc2vec', custom_transformers.DmDbowTestVectorizer(
                                    dm_path='saved_doc2vec_models2/Doc2Vec(dbow,d100,n5,hs,mc2,s0.001)',
                                    dbow_path='saved_doc2vec_models2/Doc2Vec(dmc,d100,n5,hs,w4,mc2,s0.001)',
                                )),
            ])),
            ('tfidf', Pipeline([
                ('tfidf_pipeline', custom_transformers.TfidfSavedVectorizer(
                                    tfidf_path='saved_tfidf_models/tfidf(0.3,1.0,100).pkl',
                                   )),
            ])),
            ('price_history', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
            ])),        
        ],
        transformer_weights={
            'linguistic': 1.0,
            'tfidf': 1.0,
            'price_history': 1.0
        },
    )),
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',
                nthread=4,)),
]

estimators_dm_dbow_tfidf_xgb_2 = [
    ('union', FeatureUnion(
        transformer_list=[
            ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
                ('doc2vec', custom_transformers.DmDbowTestVectorizer(
                                    dm_path='saved_doc2vec_models2/Doc2Vec(dbow,d100,n5,hs,mc2,s0.001)',
                                    dbow_path='saved_doc2vec_models2/Doc2Vec(dmc,d100,n5,hs,w3,mc2,s0.001)',
                                )),
            ])),
            ('tfidf', Pipeline([
                ('tfidf_pipeline', custom_transformers.TfidfSavedVectorizer(
                                    tfidf_path='saved_tfidf_models/tfidf(0.3,0.9,100).pkl',
                                   )),
            ])),
            ('price_history', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
            ])),        
        ],
        transformer_weights={
            'linguistic': 1.0,
            'tfidf': 1.0,
            'price_history': 1.0
        },
    )),
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',
                nthread=4,)),
]

estimators_dm_dbow_tfidf_xgb_3 = [
    ('union', FeatureUnion(
        transformer_list=[
            ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
                ('doc2vec', custom_transformers.DmDbowTestVectorizer(
                                    dm_path='saved_doc2vec_models2/Doc2Vec(dbow,d100,n5,hs,mc2,s0.001)',
                                    dbow_path='saved_doc2vec_models2/Doc2Vec(dmc,d100,n5,hs,w6,mc2,s0.001)',
                                )),
            ])),
            ('tfidf', Pipeline([
                ('tfidf_pipeline', custom_transformers.TfidfSavedVectorizer(
                                    tfidf_path='saved_tfidf_models/tfidf(0.3,0.9,100).pkl',
                                   )),
            ])),
            ('price_history', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
            ])),        
        ],
        transformer_weights={
            'linguistic': 1.0,
            'tfidf': 1.0,
            'price_history': 1.0
        },
    )),
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',
                nthread=4,)),
]


estimators_dm_dbow_tfidf_xgb_4 = [
    ('union', FeatureUnion(
        transformer_list=[
            ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
                ('doc2vec', custom_transformers.DmDbowTestVectorizer(
                                    dm_path='saved_doc2vec_models2/Doc2Vec(dbow,d100,n5,hs,mc2,s0.001)',
                                    dbow_path='saved_doc2vec_models2/Doc2Vec(dmc,d100,n5,hs,w5,mc2,s0.001)',
                                )),
            ])),
            ('tfidf', Pipeline([
                ('tfidf_pipeline', custom_transformers.TfidfSavedVectorizer(
                                    tfidf_path='saved_tfidf_models/tfidf(0.3,0.9,100).pkl',
                                   )),
            ])),
            ('price_history', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
            ])),        
        ],
        transformer_weights={
            'linguistic': 1.0,
            'tfidf': 1.0,
            'price_history': 1.0
        },
    )),
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',
                nthread=4,)),
]

param_grid_dm_dbow_tfidf_xgb = dict(
    clf__max_depth=np.array([4]),
    clf__min_child_weight=np.array([6]),
    clf__subsample=np.array([0.6]),
    clf__colsample_bytree=np.array([0.7]),
)

"""
param_grid_dm_dbow_tfidf_xgb = dict(
    clf__max_depth=np.array([3,4,5,6,7,8,9]),
    clf__min_child_weight=np.array([3,4,5,6,7,8,9]),
    clf__subsample=np.array([0.5,0.6,0.7,0.8,0.9]),
    clf__colsample_bytree=np.array([0.5,0.6,0.7,0.8,0.9]),
)
"""

estimators_ensemble_tfidf_1 = [
    ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
                ('tfidf', TfidfVectorizer()), 
                ('nmf', NMF()),
            ])),
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',
                n_jobs=1,
                nthread=1,)),
]

estimators_ensemble_tfidf_1 = [
    ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
                ('tfidf', TfidfVectorizer(sublinear_tf=True)), 
                ('nmf', NMF()),
            ])),
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',
                n_jobs=1,
                nthread=1,)),
]

estimators_ensemble_dm_dbow_1 = [
    ('linguistic', Pipeline([
                ('selector', custom_transformers.CustomDictVectorizer(key='corpus')),
                ('doc2vec', custom_transformers.Doc2VecTransformer()),
            ])),
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',
                n_jobs=1,
                nthread=1,)),
]

estimators_momentum_1 = [
    ('price_history', Pipeline([
        ('selector', custom_transformers.CustomDictVectorizer(key='price_history')),
    ])),
    ('clf', XGBClassifier(
                learning_rate = 0.1,
                n_estimators = 300,
                silent = True,
                objective = 'multi:softmax',
                n_jobs=1,
                nthread=1,)),
]
                            
ensemble_clf = [('voting', VotingClassifier(estimators=[('tfidf', Pipeline(estimators_ensemble_tfidf_1)), ('dm_dbow', Pipeline(estimators_ensemble_dm_dbow_1)), ('momentum', Pipeline(estimators_momentum_1))], voting='soft'))]

ensemble_param_grid = dict(
    voting__tfidf__linguistic__tfidf__max_df=np.array([0.9]),
    voting__tfidf__linguistic__tfidf__min_df=np.array([0.3]),
    voting__tfidf__linguistic__nmf__n_components=np.array([100]),
    voting__tfidf__clf__max_depth=np.array([4]),
    voting__tfidf__clf__min_child_weight=np.array([6]),
    voting__tfidf__clf__subsample=np.array([0.6]),
    voting__tfidf__clf__colsample_bytree=np.array([0.7]),
    voting__momentum__clf__max_depth=np.array([4]),
    voting__momentum__clf__min_child_weight=np.array([3]),
    voting__momentum__clf__subsample=np.array([0.8]),
    voting__momentum__clf__colsample_bytree=np.array([0.8]),
    voting__dm_dbow__clf__max_depth=np.array([4]),
    voting__dm_dbow__clf__min_child_weight=np.array([6]),
    voting__dm_dbow__clf__subsample=np.array([0.8]),
    voting__dm_dbow__clf__colsample_bytree=np.array([0.9]),
)
"""
param_grid_tfidf_nmf_prices_xgb = dict(
    clf__max_depth=np.array([3,4,5,6,7,8,9]),
    clf__min_child_weight=np.array([3,4,5,6,7,8,9]),
    clf__subsample=np.array([0.6,0.7,0.8,0.9]),
    clf__colsample_bytree=np.array([0.6,0.7,0.8,0.9]),
    union__linguistic__tfidf__max_df=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    union__linguistic__tfidf__min_df=np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
    union__linguistic__tfidf__sublinear_tf=[True],
    union__linguistic__nmf__n_components=np.array([50, 100, 200, 300, 400, 500, 600]),
)
"""
