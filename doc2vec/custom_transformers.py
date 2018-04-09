from gensim.models import Doc2Vec
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import joblib

"""http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
"""

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
        
class CustomDictVectorizer(BaseEstimator, TransformerMixin):
    """ ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """Transform feature->value dicts to array or sparse matrix.
        Named features not encountered during fit or fit_transform will be
        silently ignored.
        Parameters
        ----------
        X : Mapping or iterable over Mappings, length = n_samples
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        if self.key == 'corpus':
            Xa = np.empty([len(X)], dtype=object)
            for i, x in enumerate(X):
                Xa[i] = x['corpus']
            """Memory error at this line for 10k docs (Xa = np.array(Xa))
            https://stackoverflow.com/questions/39032200/converting-python-list-to-numpy-array-inplace
            """
        elif self.key == 'price_history':
            Xa = np.zeros((len(X), 5)) # 2-d array of price histories
            for i, x in enumerate(X):
                for j, val in enumerate(x['price_history']):
                    Xa[i][j] = val
        elif self.key == 'doc2vec':
            Xa = np.empty([len(X), 100])
            for i, x in enumerate(X):
                Xa[i] = x['corpus']
        return Xa

import time
import gensim
class DmDbowTrainVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dm_path, dbow_path):
        """NOTE: we remove this from the pipeline because it doesn't make sense to infer_vector
        thousands of times for the gridsearch. we move this computation to preprocessing.
        args :
            model_path : e.g., 'saved_doc2vec_models/Doc2Vec(dmm,d100,n5,w10,mc2,s0.001,t8)'
        """
        start = time.time()
        self.dm_path = dm_path
        self.dbow_path = dbow_path
        # saved_doc2vec_models2/Doc2Vec(dbow,d100,n5,hs,mc2,s0.001)
        # 012345678901234567890123456789012345678
        # saved_doc2vec_models2/Doc2Vec(dmc,d100,n5,hs,w3,mc2,s0.001)
        # 01234567890123456789012345678901234567
        self.dm_size = int(dm_path[36:39])
        self.dbow_size = int(dbow_path[35:38])
        self.dm = Doc2Vec.load(dm_path)
        self.dbow = Doc2Vec.load(dbow_path)
        assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
        end = time.time()
        print("dm_dbow __init__ took {} seconds".format(end-start))

    def fit(self, x, y=None):
        #print("dm_dbow fit was called")
        return self

    def transform(self, X):
        #start = time.time()
        Xa = np.zeros((len(X), self.dm_size+self.dbow_size))
        # TODO: This seems to not be vectorizable because
        # passed in X's could be randomized and not contiguous
        for i in range(np.shape(X)[0]): 
            Xa[:, :self.dm_size] = self.dm.docvecs[X[:][0]]
            Xa[:, self.dm_size:] = self.dbow.docvecs[X[:][0]]
        #end = time.time()
        print("dm_dbow train transform took {} seconds".format(end-start))
        return Xa

class DmDbowTestVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dm_path, dbow_path):
        """NOTE: we remove this from the pipeline because it doesn't make sense to infer_vector
        thousands of times for the gridsearch. we move this computation to preprocessing.
        args :
            model_path : e.g., 'saved_doc2vec_models/Doc2Vec(dmm,d100,n5,w10,mc2,s0.001,t8)'
        """
        start = time.time()
        self.dm_path = dm_path
        self.dbow_path = dbow_path
        # saved_doc2vec_models2/Doc2Vec(dbow,d100,n5,hs,mc2,s0.001)
        # 012345678901234567890123456789012345678
        # saved_doc2vec_models2/Doc2Vec(dmc,d100,n5,hs,w3,mc2,s0.001)
        # 01234567890123456789012345678901234567
        self.dm_size = int(dm_path[36:39])
        self.dbow_size = int(dbow_path[35:38])
        self.dm = Doc2Vec.load(dm_path)
        self.dbow = Doc2Vec.load(dbow_path)
        assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
        end = time.time()
        print("dm_dbow __init__ took {} seconds".format(end-start))

    def fit(self, x, y=None):
        print("dm_dbow fit was called")
        return self

    def transform(self, X):
        start = time.time()
        Xa = np.zeros((len(X), self.dm_size+self.dbow_size))
        for i in range(np.shape(X)[0]):
            Xa[i, :self.dm_size] = self.dm.infer_vector(X[i])
            Xa[i, self.dm_size:] = self.dbow.infer_vector(X[i])
        end = time.time()
        print("dm_dbow transform took {} seconds".format(end-start))
        return Xa
    
class TfidfSavedVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf_path):
        self.tfidf_path = tfidf_path
        print(self.tfidf_path)
        """NOTE: we remove this from the pipeline because it doesn't make sense to infer_vector
        thousands of times for the gridsearch. we move this computation to preprocessing.
        args :
            model_path : e.g., 'saved_doc2vec_models/Doc2Vec(dmm,d100,n5,w10,mc2,s0.001,t8)'
        """
        start = time.time()
        # saved_tfidf_models/tfidf(0.2,0.8,300).pkl
        # 012345678901234567890123456789012345
        self.tfidf_size = int(self.tfidf_path[33:36])
        with open(tfidf_path, 'rb') as p:
            self.tfidf = joblib.load(p)
        end = time.time()
        print("tfidf __init__ took {} seconds".format(end-start))

    def fit(self, x, y=None):
        print("tfidf fit was called")
        return self

    def transform(self, X):
        start = time.time()
        Xa = np.zeros((len(X), self.tfidf_size))
        Xa = self.tfidf.transform(X)
        end = time.time()
        print("tfidf transform took {} seconds".format(end-start))
        return Xa
    
import os.path
class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vector_size = 100
        # Hyperparameter choices influenced by http://www.aclweb.org/anthology/W16-1609
        self.dm = Doc2Vec(dm=1, dm_mean=1, size=self.vector_size, window=5, max_vocab_size=100000, negative=5, hs=1, min_count=5, workers=24, alpha=0.025, min_alpha=0.0001, sample=0.000001)
        self.dbow = model = Doc2Vec(dm=0, size=self.vector_size, window=15, max_vocab_size=100000, negative=5, hs=1, min_count=5, workers=24, alpha=0.025, min_alpha=0.0001, sample=0.000001)
        
    def fit(self, X, y=None):
        start = time.time()
        dm = self.dm
        dbow = self.dbow
        dm_path = 'saved_doc2vec_models3/' + str(dm).replace('/','')
        dbow_path = 'saved_doc2vec_models3/' + str(dbow).replace('/','')
        if (os.path.isfile(dm_path) and os.path.isfile(dbow_path)):
            print("Using pretrained model")
            dm = Doc2Vec.load(dm_path)
            dbow = Doc2Vec.load(dbow_path)
        else:
            print("Training Doc2Vec model")
            Xa = np.empty([len(X)], dtype=object)
            for i, x in enumerate(X):
                tags = [i]
                Xa[i] = gensim.models.doc2vec.TaggedDocument(x, tags)
            X = Xa
            dm.build_vocab(X)
            dbow.reset_from(dm)
            dm.train(X, total_examples=len(X), epochs=600)
            dbow.train(X, total_examples=len(X), epochs=20)
            dm.save(dm_path)
            dbow.save(dbow_path)
        self.dm = dm
        self.dbow = dbow
        end = time.time()
        print("Doc2VecTransformer fit took {0}".format(end-start))
        return self
    
    def transform(self, X):
        Xa = np.zeros((len(X), self.vector_size*2))
        for i in range(len(X)):
            Xa[i, :self.vector_size] = self.dm.infer_vector(X[i])
            Xa[i, self.vector_size:] = self.dbow.infer_vector(X[i])
        return Xa
    
class TfidfNMFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, min_df, max_df, n_components):
        key = 'tfidf({:02.1f},{:02.1f},{{:03d}})'.format(min_df, max_df, n_components) + '.pkl'
        # tfidf(0.2,0.7,300)
        # 01234567890123456
        # self.vector_size = int(key[14:17])
        pickle_path = 'saved_tfidf_nmf_models/' + key
        with open(pickle_path):
            self.model = pickle.loads(pickle_path)
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        Xa = self.model.transform(X)
        return Xa
        
class DebugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        print("mean of all X is {}".format(np.mean(X.flatten())))
        print("std of all X is {}".format(np.std(X.flatten())))
        print("column means of X is {}".format(np.mean(X, axis=0)))
        print("column stds of X is {}".format(np.std(X, axis=0)))
        return X
        
