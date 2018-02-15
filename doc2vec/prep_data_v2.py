import itertools
import time
import os
import spacy
import utils_v2
import my_config


def main():
    print("Process started...")
    start = time.time()
    
    preprocess_texts()
    prepare_dataset()
    
    end = time.time()
    print("Total running time: {0}".format(end-start))
    
def preprocess_texts():
    """https://spacy.io/usage/processing-pipelines#section-pipelines
    """
    nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])  #  disable parser, named entity recognizer
                                                       # lemmatization occurs in the tagger (?)
                                                       # https://spacy.io/usage/linguistic-features#section-pos-tagging

    """https://github.com/explosion/spacy/issues/172#issuecomment-183963403
    """
    gen1, gen2 = itertools.tee(utils_v2.load_texts(my_config.source_dir, 'train', my_config.train_quarters, [], yield_paths=True))
    texts = (text for (text, file_path) in gen1)
    file_paths = (file_path for (text, file_path) in gen2)
    
    """https://spacy.io/usage/processing-pipelines#section-multithreading
    """
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=8): # yields Doc objects in order
        output_file = next(file_paths).replace(my_config.source_dir, my_config.output_dir)
        print(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as t:
            t.write(" ".join(list(map(lambda x: x.lemma_, filter(lambda token: token.is_alpha and not token.is_stop and not token.is_oov, doc)))))
            
def prepare_dataset():
    """ Prepares a nice dataset for analysis.
    """
    gen = utils_v2.load_data(my_config.data_dir, 'train', my_config.train_quarters, my_config.test_quarters, yield_paths=True) 
    data, labels, file_paths = utils_v2.split_gen_3(gen)
    corpus, price_history = utils_v2.split_gen(data)
    alpha1, alpha2, alpha3, alpha, alpha5 = utils_v2.split_gen_5(labels)
    data_index_path = '../data/' + my_config.dataset_dir + '/train.txt'
    open(data_index_path, 'w')  # clear file
    with open(data_index_path, 'a') as data_index:
        for doc in corpus:
            output_file = next(file_paths).replace(my_config.source_dir, my_config.dataset_dir)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as t:
                t.write(doc)
        prices = next(price_history)
        data_index.write('{};{},{},{},{},{};{},{},{},{},{}'.format(output_file, prices[0], prices[1], prices[2], prices[3], price[4], alpha1, alpha2, alpha3, alpha4, alpha5))
    
if __name__ == "__main__":
    main()
