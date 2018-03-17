from gensim.models import Doc2Vec
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

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
        return Xa

class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """NOTE: we remove this from the pipeline because it doesn't make sense to infer_vector
        thousands of times for the gridsearch. we move this computation to preprocessing.
        args :
            model_path : e.g., 'saved_doc2vec_models/Doc2Vec(dmm,d100,n5,w10,mc2,s0.001,t8)'
        """
        model_pth = 'saved_doc2vec_models/Doc2Vec(dmc,d100,n5,w5,mc2,s0.001,t8)' # TODO: Why can't I pass this in??
        self.model = Doc2Vec.load(model_pth)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        Xa = np.zeros((len(X), 100))
        for i in range(np.shape(X)[0]):
            Xa[i, :] = self.model.infer_vector(X[i])
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
        
