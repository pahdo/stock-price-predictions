import itertools
import time
import os
import sys
import spacy
from spacy.lang.en.lemmatizer import LOOKUP
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token
import my_config
import my_diagnostics
import utils_v2

def main():
    print("Process started...")
    start = time.time()
    
    """Using Spacy tagger - 35000 seconds/9 hours to run. 52GB RAM usage, 140GB read, 400GB write
    7238 files -> 7001 files - preprocess texts
    2017/QTR1 2.41GB -> 1.43GB
    """
    """Using custom en_lemmatizer - 22125 seconds/6.1 hours to run - preprocess texts
    7734 items -> 7734 items
    2017/QTR2 1.31GB -> 270.2MB
    """
    start_qtr_num = int(sys.argv[1])
    end_qtr_num = int(sys.argv[2])
    split = sys.argv[3]
    if split == 'train':
        quarters = my_config.train_quarters
    elif split == 'test':
        quarters = my_config.test_quarters
    else:
        print('please specify test or train')
        exit()

    print("max quarter num is {}".format(len(quarters)-1))

    for i in range(start_qtr_num, end_qtr_num+1):
        preprocess_texts(quarters[i], i)
        prepare_dataset(quarters[i], i, split)

    end = time.time()
    print("total running time: {0}".format(end-start))
    
def preprocess_texts(train_qtrs, part_num):
    """https://spacy.io/usage/models
    spacy default pipeline tagger - memory error
    spacy blank model
    """
    nlp = spacy.blank('en')

    """https://github.com/explosion/spaCy/issues/1154
    lookup-based lemmatization
    """
    def en_lemmatizer(doc):
        for token in doc:
            token.lemma_ = LOOKUP.get(token.lower_, u'oov')
            if token in STOP_WORDS:
                token.is_stop = True
        return doc

    """https://explosion.ai/blog/spacy-v2-pipelines-extensions
    custom pipelines in Spacy 2.0
    """
    nlp.add_pipe(en_lemmatizer, name='en_lemmatizer', last=True)

    """https://github.com/explosion/spacy/issues/172#issuecomment-183963403
    how to split generators
    """
    gen1, gen2 = itertools.tee(utils_v2.load_texts(my_config.source_dir, train_qtrs))
    texts = (text for (text, file_path) in gen1)
    file_paths = (file_path for (text, file_path) in gen2)
    
    """https://spacy.io/usage/processing-pipelines#section-multithreading
    """
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=64):
        output_file = next(file_paths).replace(my_config.source_dir, my_config.output_dir)
        os.makedirs(os.path.dirname(output_file), exist_ok=True) # redundant calls take ~ 0.00029206275939941406 seconds each
        with open(output_file, 'w') as t:
            t.write(" ".join(list(map(lambda x: x.lemma_, filter(lambda token: token.is_alpha and not token.is_stop and not token.lemma_ == u'oov', doc)))))
        print("wrote {}".format(output_file))
    
            
def prepare_dataset(train_qtrs, part_num, split):
    """ Running time: 124 seconds on 7001 documents, producing 1349 lines (19% of data has matching price information)
    """
    gen = utils_v2.load_data(my_config.data_dir, 'subset', train_qtrs, yield_paths=True) 
    data, labels, file_paths = utils_v2.split_gen_3(gen)
    corpus, price_history = utils_v2.split_gen(data)
    alpha1, alpha2, alpha3, alpha4, alpha5 = utils_v2.split_gen_5(labels)
    data_index_path = 'data/' + my_config.dataset_dir + '/' + split + str(part_num) + '.txt'
    os.makedirs(os.path.dirname(data_index_path), exist_ok=True)
    open(data_index_path, 'w+')  # clear file
    with open(data_index_path, 'a') as data_index:
        for doc in corpus:
            output_file = next(file_paths).replace(my_config.source_dir, my_config.dataset_dir)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as t: # Copy the file AGAIN because we should only put data with labels in dataset dir
                t.write(doc)
            prices = next(price_history)
            data_index.write('{};{},{},{},{},{};{},{},{},{},{}\n'.format(output_file, prices[0], prices[1], prices[2], prices[3], prices[4], next(alpha1), next(alpha2), next(alpha3), next(alpha4), next(alpha5)))
    
if __name__ == "__main__":
    main()
