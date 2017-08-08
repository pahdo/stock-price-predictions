### Spacy Prep Data - Attempts to stem, remove OOV, and remove stop words but too expensive to run (16GB of RAM, runtime is high too)
#Here, we call NLP on my entire corpus of text data, but it uses too much memory and takes too long to compute

def prepData():
    folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg'] 
    print("Preparing dataset...")
    alldata = u''
    for fol in folders:
        temp = u''
        txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
        i = 0
        for txt in txt_files:
            with smart_open.smart_open(txt, "rb") as t:
#               #start = time.clock()
                #doc = nlp(t.read().decode("utf-8")) # Constructing an nlp object tokenizes and stems automatically
#                 doc = nlp(t.read().decode("utf-8"))
#                 end = time.clock()
#                 print ("doc = nlp.make_doc(t.read().decode(\"utf-8\")): ", end-start)
                
#                 start = time.clock()
#                 removed_stop_words = list(map(lambda x: x.lemma_, filter(lambda token: token.is_alpha and not token.is_stop and not token.is_oov, doc)))
#                 end = time.clock()
#                 print ("removed_stop_words = list(filter(lambda token: lambda token: ", end-start)
#                 break
                
                #start = time.clock()
                #temp += " ".join(removed_stop_words)
                #end = time.clock()
                #print ("temp += " ".join(removed_stop_words): ", end-start)
                temp += t.read().decode("utf-8")
            temp += '\n'
        output = fol.replace('/', '-') + '.txt'
        print("{} aggregated".format(os.path.join(dirname, output)))
        with smart_open.smart_open(os.path.join(dirname, output), "wb") as f:
            for idx, line in enumerate(temp.split('\n')):
                num_line = u"_*{0} {1}\n".format(idx, line)
                f.write(num_line.encode("UTF-8"))
        alldata += temp
    with smart_open.smart_open(os.path.join(dirname, 'alldata-id.txt'), 'wb') as f:
        for idx, line in enumerate(alldata.split('\n')):
            #num_line = u"_*{0} {1}\n".format(idx, line)
            num_line = line
            f.write(num_line.encode("UTF-8")) 
    print("alldata-id.txt aggregated")
    return alldata


start = time.clock()
prepData()
print("Ok...")
end = time.clock()
print ("Total running time: ", end-start)
with smart_open.smart_open(os.path.join(dirname, 'alldata-id.txt'), "rb") as t:
    print("Opened the doc")
    start = time.clock()
    doc = nlp(t.read().decode("utf-8"))
    end = time.clock()
    print ("doc = nlp(t.read().decode(\"utf-8\")): ", end-start)
    removed_stop_words = list(map(lambda x: x.lemma_, filter(lambda token: token.is_alpha and not token.is_stop and not token.is_oov, doc)))
    with smart_open.smart_open(os.path.join(dirname, 'alldata-id-clean.txt'), "wb") as f:
        for idx, line in enumerate(" ".join(removed_stop_words).split('\n')):
            num_line = u"_*{0} {1}\n".format(idx, line)
            f.write(num_line.encode("UTF-8")) 