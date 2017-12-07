"""
prep_data.py cleans SEC Forms and creates files of the form aggregated-pos-###.txt and aggregated-neg-###.txt
    from data/pos and data/neg, where each line is formatted as [doc_id] [sentiment] [words ...].
"""

###### CONFIGURATION ######
dirname = 'data_by_returns_small'

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
        doc = nlp.make_doc(t.read().decode("utf-8"))
        """
        Process an individual text document.
        The first 500 words in a SEC Form are considered the "header" and removed. The words are converted to lowercase.
        Non-alphabetic words, stop words, and words outside of the English vocabulary are removed.
        """
        removed_stop_words = list(map(lambda x: x.lower_, filter(lambda token: token.is_alpha and not token.is_stop and not token.is_oov, doc)))[500:]
        return " ".join(removed_stop_words)

def prepData():
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

aggregate_data('aggregated-*.txt', 'alldata-id.txt')
aggregate_data('aggregated-pos-*.txt', 'pos-all.txt')
aggregate_data('aggregated-neg-*.txt', 'neg-all.txt')

print("Processed completed")
end = time.time()
print("Total running time: {0}".format(end-start))
