### Spacy Prep Data - Remove OOV, remove stop words

def prepData():
    folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg'] 
    print("Preparing dataset...")
    alldata = u''
    for fol in folders:
        temp = u''
        txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
        for txt in txt_files:
            with smart_open.smart_open(txt, "rb") as t:
                doc = nlp.make_doc(t.read().decode("utf-8")) # Constructing an nlp object tokenizes and stems automatically
                removed_stop_words = list(map(lambda x: x.lower_, filter(lambda token: token.is_alpha and not token.is_stop and not token.is_oov, doc)))   
                temp += " ".join(removed_stop_words)
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
            num_line = u"_*{0} {1}\n".format(idx, line)
            #num_line = line
            f.write(num_line.encode("UTF-8")) 
    print("alldata-id.txt aggregated")
    return alldata

start = time.clock()
alldata = prepData()
end = time.clock()
print ("Total running time: ", end-start)