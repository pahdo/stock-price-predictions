import itertools
import time
import os
import spacy
import my_config
import utils_v2


def main():
    print("Process started...")
    start = time.time()
    
    """NOTE TO SELF: next time you run this, you already ran preprocess_texts() on 2017/QTR1, so just run prepare_dataset
    ^^^ The above took 35000 seconds or 9 hours to run and used 52GB of ram with 140GB read and 400GB write
    I also somehow went from 7238 files to 7001 files
    """

    preprocess_texts()
    prepare_dataset()
    
    end = time.time()
    print("Total running time: {0}".format(end-start))
    
def preprocess_texts():
    """https://spacy.io/usage/processing-pipelines#section-pipelines
    default spacy pipeline contains [tagger, parser, named entity recognizer]
    """
    """https://github.com/explosion/spaCy/issues/1154
    https://spacy.io/usage/linguistic-features#section-pos-tagging
    lemmatization occurs in the tagger
    """
    nlp = spacy.load('en', disable=['parser', 'ner'])

    """https://github.com/explosion/spacy/issues/172#issuecomment-183963403
    how to split generators
    """
    gen1, gen2 = itertools.tee(utils_v2.load_texts(my_config.source_dir, 'train', my_config.train_quarters, []))
    texts = (text for (text, file_path) in gen1)
    file_paths = (file_path for (text, file_path) in gen2)
    
    """https://spacy.io/usage/processing-pipelines#section-multithreading
    """
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=8): # yields Doc objects in order
        output_file = next(file_paths).replace(my_config.source_dir, my_config.output_dir)
        print(output_file)
        """TODO: You do this thousands of times, but maybe you should only do it once per directory
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as t:
            t.write(" ".join(list(map(lambda x: x.lemma_, filter(lambda token: token.is_alpha and not token.is_stop and not token.is_oov, doc)))))
            
def prepare_dataset():
    """ Prepares a nice dataset for analysis.
    """
    """ Running time: 124 seconds on 7001 documents, producing 1349 lines (19% of data has matching price information)
    """
    gen = utils_v2.load_data(my_config.data_dir, 'train', my_config.train_quarters, my_config.test_quarters, yield_paths=True) 
    data, labels, file_paths = utils_v2.split_gen_3(gen)
    corpus, price_history = utils_v2.split_gen(data)
    alpha1, alpha2, alpha3, alpha4, alpha5 = utils_v2.split_gen_5(labels)
    data_index_path = 'data/' + my_config.dataset_dir + '/train.txt'
    os.makedirs(os.path.dirname(data_index_path), exist_ok=True)
    open(data_index_path, 'w+')  # clear file
    with open(data_index_path, 'a') as data_index:
        for doc in corpus:
            output_file = next(file_paths).replace(my_config.source_dir, my_config.dataset_dir)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as t:
                t.write(doc)
            prices = next(price_history)
            data_index.write('{};{},{},{},{},{};{},{},{},{},{}\n'.format(output_file, prices[0], prices[1], prices[2], prices[3], prices[4], next(alpha1), next(alpha2), next(alpha3), next(alpha4), next(alpha5)))
    
if __name__ == "__main__":
    main()
