from multiprocessing import Pool
import smart_open
import os.path
import spacy
import time
import glob

nlp = spacy.load('en')
dirname = 'data'

print("process started")

def aggregate_data(name, out):
    txt_files = glob.glob(os.path.join(dirname, name))
    open(os.path.join(dirname, out), 'w').close() # Clear file
    print(len(txt_files))
    with smart_open.smart_open(os.path.join(dirname, out), 'ab') as f:
        for txt in txt_files:
            for line in open(txt, 'rb'):
                num_line = u"{0}\n".format(line)
                f.write(num_line.encode("UTF-8"))
    print("{0} aggregated".format(out))
    
aggregate_data('aggregated-*.txt', 'alldata-id.txt')