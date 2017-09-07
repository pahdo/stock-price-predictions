from collections import namedtuple
import gensim.utils
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
def read_labeled_corpus(fpos, fneg, split):
    f_list = [fpos, fneg]
    i = 0
    for f in f_list:
        if i == 0:
            sentiment = 1.0
        else:
            sentiment = 0.0
        for line in open(f, encoding='utf-8'):
            tokens = gensim.utils.to_unicode(line).split()
            if(len(tokens)==0):
                continue
            words = tokens[1:]
            tags = [tokens[0]]
            yield SentimentDocument(gensim.utils.to_unicode(line).split()[1:], tags, split, sentiment)
        i+=1