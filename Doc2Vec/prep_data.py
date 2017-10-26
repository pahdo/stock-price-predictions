"""
New behavior: prep_data.py cleans SEC Forms, creates files of the form aggregated-pos.txt and aggregated-neg.txt
    from data/pos and data/neg, where each line is formatted as [doc_id] [sentiment] [words ...]
    This makes it easier to shuffle the data set for cross-validation. In this new behavior, we can read in strings
    of format [doc_id] [sentiment] [words ...] to create tuples of format (sentiment, model[doc_id])
Old behavior: prep_data.py cleans SEC Forms, creates files of the form aggregated-train-pos*.txt, etc., and aggregates
    them into files train-pos.txt, etc.
"""

###### CONFIGURATION ######
dirname = 'data_by_returns'

###########################

from multiprocessing import Pool
import smart_open
import os.path
import spacy
import time
import glob
nlp = spacy.load('en')

print("Process started...")
start = time.time()

def processOne(txt):
    with smart_open.smart_open(txt, "rb") as t:
        """
        # New behavior: slower proprocessing with lemmatization - may take 114000 seconds (100 days)
        process_one_start = time.time()
        doc = nlp(t.read().decode("utf-8"))
        process_one_end = time.time()
        print("Process one time: {0}".format(process_one_end-process_one_start))
        removed_stop_words = list(map(lambda x: x.lemma_, filter(lambda token: token.is_alpha and not token.is_stop and not token.is_oov, doc)))[500:]
        """
        """
        # Sike this still replaces everything with blanks or "because"
        #process_one_start = time.time()
        doc = nlp.make_doc(t.read().decode("utf-8"))
        #process_one_end = time.time()
        #print("Process one time: {0}".format(process_one_end-process_one_start))
        removed_stop_words = list(map(lambda x: x.lemma_, filter(lambda token: token.is_alpha and not token.is_stop and not token.is_oov, doc)))[500:]
        """
        # New old behavior: Faster preprocessing without lemmatization
        doc = nlp.make_doc(t.read().decode("utf-8"))
        # Approximately top 500 words in a SEC Form are header
        removed_stop_words = list(map(lambda x: x.lower_, filter(lambda token: token.is_alpha and not token.is_stop and not token.is_oov, doc)))[500:]
        return " ".join(removed_stop_words)

def prepData():
    """
    # Old behavior: prep_data.py cleans SEC Forms, creates files of the form aggregated-train-pos*.txt, etc., and aggregates
    # them into files train-pos.txt, etc.
    folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'valid/pos', 'valid/neg']
    """
    folders = ['pos', 'neg']
    print("Preparing dataset...")
    pool = Pool()
    num_processed = 0
    batch_size = 200
    for fol in folders:
        temp = u''
        txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
        print("Processing {0} files, {1} at a time in {2}".format(len(txt_files), batch_size, fol))
        for i in range(0, len(txt_files), batch_size):
            if (i%1000==0):
                print("Finished processing {0} files in memory, aggregated {1} total files so far".format(i, num_processed))
            if (i+200 > len(txt_files)):
                end = len(txt_files)
            else:
                end = i+200
            results = pool.map(processOne, txt_files[i:end])
            temp += '\n'.join(results)
            temp += '\n'
            if (end % 5000==0 or end==len(txt_files)):
                output = 'aggregated-{0}-{1}.txt'.format(fol.replace('/', '-'), num_processed)
                last_idx = 0
                with smart_open.smart_open(os.path.join(dirname, output), "wb") as f:
                    for idx, line in enumerate(temp.split('\n')):
                        num_line = u"{0} {1} {2}\n".format(num_processed+idx, '1' if fol=='pos' else '0', line)
                        f.write(num_line.encode('UTF-8'))
                        last_idx = idx
                num_processed += last_idx
                temp = u''
                print("{} aggregated".format(os.path.join(dirname, output)))
        
prepData()

def aggregate_data(name, out):
    txt_files = glob.glob(os.path.join(dirname, name))
    open(os.path.join(dirname, out), 'wb').close() # Clear file
    print(len(txt_files))
    with open(os.path.join(dirname, out), 'ab') as f:
        for txt in txt_files:
            for line in open(txt, 'r'):
                f.write('{}'.format(line).encode('UTF-8'))
    print("{0} aggregated".format(out))

"""    
# Old behavior: prep_data.py cleans SEC Forms, creates files of the form aggregated-train-pos*.txt, etc., and aggregates
# them into files train-pos.txt, etc.
aggregate_data('aggregated-*.txt', 'alldata-id.txt')
aggregate_data('aggregated-train*.txt', 'train-all.txt')
aggregate_data('aggregated-test*.txt', 'test-all.txt')
aggregate_data('aggregated-train-pos*.txt', 'train-pos.txt')
aggregate_data('aggregated-train-neg*.txt', 'train-neg.txt')
aggregate_data('aggregated-test-pos*.txt', 'test-pos.txt')
aggregate_data('aggregated-test-neg*.txt', 'test-neg.txt')
aggregate_data('aggregated-valid-pos*.txt', 'valid-pos.txt')
aggregate_data('aggregated-valid-neg*.txt', 'valid-neg.txt')
"""

aggregate_data('aggregated-*.txt', 'alldata-id.txt')
aggregate_data('aggregated-pos-*.txt', 'pos-all.txt')
aggregate_data('aggregated-neg-*.txt', 'neg-all.txt')

print("Processed completed")
end = time.time()
print("Total running time: {0}".format(end-start))
