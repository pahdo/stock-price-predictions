import itertools
import time
import os
import spacy
import utils_v2

###### CONFIGURATION ######
train_quarters = [
    '2005/QTR1', '2005/QTR2', '2005/QTR3', '2005/QTR4',
    '2004/QTR1', '2004/QTR2', '2004/QTR3', '2004/QTR4',
    '2003/QTR1', '2003/QTR2', '2003/QTR3', '2003/QTR4',
    '2002/QTR1', '2002/QTR2', '2002/QTR3', '2002/QTR4',
    '2001/QTR1', '2001/QTR2', '2001/QTR3', '2001/QTR4',
    '2000/QTR1', '2000/QTR2', '2000/QTR3', '2000/QTR4',
    '1999/QTR1', '1999/QTR2', '1999/QTR3', '1999/QTR4',
    '1998/QTR1', '1998/QTR2', '1998/QTR3', '1998/QTR4',
    '1997/QTR1', '1997/QTR2', '1997/QTR3', '1997/QTR4',
    '1996/QTR1', '1996/QTR2', '1996/QTR3', '1996/QTR4',
    '1995/QTR1', '1995/QTR2', '1995/QTR3', '1995/QTR4',
    '1994/QTR1', '1994/QTR2', '1994/QTR3', '1994/QTR4',
    '2009/QTR1', '2009/QTR2', '2009/QTR3', '2009/QTR4',
    '2008/QTR1', '2008/QTR2', '2008/QTR3', '2008/QTR4',
    '2007/QTR1', '2007/QTR2', '2007/QTR3', '2007/QTR4',
    '2006/QTR1', '2006/QTR2', '2006/QTR3', '2006/QTR4']
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
    gen1, gen2 = itertools.tee(utils_v2.load_texts(source_dir, 'all', train_quarters, [], yield_paths=True))
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
