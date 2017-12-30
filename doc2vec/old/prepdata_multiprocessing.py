from multiprocessing import Pool
import smart_open
import os.path
import spacy
import time
import glob

start = time.clock()
start_time = time.time()

nlp = spacy.load('en')
dirname = 'data'

def processOne(txt):
    with smart_open.smart_open(txt, "rb") as t:
        doc = nlp(t.read().decode("utf-8")) # Constructing an nlp object tokenizes and stems automatically
        removed_stop_words = list(map(lambda x: x.lemma_, filter(lambda token: token.is_alpha and not token.is_stop and not token.is_oov, doc)))
        return " ".join(removed_stop_words)
def prepData():
    folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg'] 
    print("Preparing dataset...")
    alldata = u''
    pool = Pool()
    for fol in folders:
        temp = u''
        txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))[:40]
        results = pool.map(processOne, txt_files)
        temp += '\n'.join(results)
        temp += '\n'
        #print(results)
        output = fol.replace('/', '-') + '.txt'
        print("{} aggregated".format(os.path.join(dirname, output)))
        with smart_open.smart_open(os.path.join(dirname, output), "wb") as f:
            for idx, line in enumerate(temp.split('\n')):
                #num_line = u"_*{0} {1}\n".format(idx, line)
                num_line = u"{0}\n".format(line)
                f.write(num_line.encode("UTF-8"))
        alldata += temp
    with smart_open.smart_open(os.path.join(dirname, 'alldata-id.txt'), 'wb') as f:
        for idx, line in enumerate(alldata.split('\n')):
            #num_line = u"_*{0} {1}\n".format(idx, line)
            num_line = u"{0}\n".format(line)
            f.write(num_line.encode("UTF-8")) 
    print("alldata-id.txt aggregated")
    return alldata

prepData()

end = time.clock()
end_time = time.time()
print ("Total processor time: ", end-start)
print ("Total running time: ", end_time-start_time)