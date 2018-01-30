import itertools
import time
import os
import spacy
import utils_v2

###### CONFIGURATION ######

source_dir = '10-X_C'
output_dir = '10-X_C_clean'

###########################

def main():
    print("Process started...")
    start = time.time()

    """https://spacy.io/usage/processing-pipelines#section-pipelines
    """
    nlp = spacy.load('en', disable=['parser', 'ner']) # disable parser, named entity recognizer
                                                    # lemmatization occurs in the tagger (?)
                                                    # https://spacy.io/usage/linguistic-features#section-pos-tagging

    """https://github.com/explosion/spacy/issues/172#issuecomment-183963403
    """
    gen1, gen2 = itertools.tee(utils_v2.load_texts(source_dir, split='all', yield_paths=True))
    #print("debug {}".format(next(gen1)))
    print(len(next(gen1)))
    texts = (text for (text, file_path) in gen1)
    file_paths = (file_path for (text, file_path) in gen2)
    #print("debug: {}".format(next(texts)))

    """https://spacy.io/usage/processing-pipelines#section-multithreading
    """
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=8): # yields Doc objects in order
        output_file = next(file_paths).replace(source_dir, output_dir)
        print(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as t:
            t.write(" ".join(list(map(lambda x: x.lemma_, filter(lambda token: token.is_alpha and not token.is_stop and not token.is_oov, doc)))))

    end = time.time()
    print("Total running time: {0}".format(end-start))

if __name__ == "__main__":
    main()
