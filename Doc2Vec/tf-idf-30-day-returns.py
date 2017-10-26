from collections import namedtuple
from Doc2VecUtils import SentimentDocument
from Doc2VecUtils import file_len
import os.path
import gensim.utils

def read_corpus_plain(fname):
    for line in open(fname, encoding='utf-8'):
        # For training data, add tags
        tokens = gensim.utils.to_unicode(line).split()
        if(len(tokens)==0):
            continue
        words = tokens[1:]
        yield ' '.join(words)
SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
def read_labeled_corpus(f):
    for line in open(f, encoding='utf-8'):
        tokens = gensim.utils.to_unicode(line).split()
        if(len(tokens)==0):
            continue
        words = tokens[2:]
        tags = [tokens[0]]
        sentiment = tokens[1]
        yield SentimentDocument(words, tags, -1, sentiment)
        
dirname = 'data_by_returns'
num_lines = file_len(os.path.join(dirname, 'alldata-id.txt'))
num_lines_pos = file_len(os.path.join(dirname, 'pos-all.txt'))
num_lines_neg = file_len(os.path.join(dirname, 'neg-all.txt'))

# SKLearn Tf-idf depends on loading the entire corpus into memory
#from sklearn.feature_extraction.text import TfidfVectorizer
#import pandas as pd

#vectorizer = TfidfVectorizer(max_features=300, max_df=0.3, min_df=0.1, sublinear_tf=True)

corpus = read_labeled_corpus(os.path.join(dirname, 'alldata-id.txt'))

temp = read_labeled_corpus(os.path.join(dirname, 'alldata-id.txt'))
labels = [doc["sentiment"] for doc in temp]

from gensim.sklearn_api import TfIdfTransformer
tfidf_model = TfIdfTransformer()
embedded = tfidf_model.fit(corpus[words])
clf = linear_model.LogisticRegression(penalty='l2', C=0.1)

corpus = read_labeled_corpus(os.path.join(dirname, 'alldata-id.txt'))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, corpus, labels, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
