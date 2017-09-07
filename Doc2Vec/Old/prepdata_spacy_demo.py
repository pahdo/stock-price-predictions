### Demo - Using Spacy to stem, remove OOV, and remove stop words

start = time.clock()
folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg'] 
print("Preparing dataset...")
alldata = u''
for fol in folders:
    temp = u''
    txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
    for txt in txt_files:
        with smart_open.smart_open(txt, "rb") as t:
            doc = nlp(t.read().decode("utf-8")) # Constructing an nlp object tokenizes and stems automatically
            tokens = [token.orth_ for token in doc]
            in_vocab = list(filter(lambda token: token.is_alpha and token.lemma_ in nlp.vocab, doc))
            removed_stop_words = list(filter(lambda token: not token.is_stop, in_vocab))
            in_vocab = [token.lemma_ for token in in_vocab]
            removed_stop_words = [token.lemma_ for token in removed_stop_words]
            print("Original length: {} \nRemoved non-words: {} \nRemoved stop words: {}"
                  .format(len(tokens), len(in_vocab), len(removed_stop_words)))
            df = pd.DataFrame(list(itertools.zip_longest(tokens, in_vocab, removed_stop_words)), 
                              columns=['tokens', 'in_vocab', 'removed_stop_words'])
            display(df)
            t_clean = " ".join(removed_stop_words)
            print(t_clean)
            break
end = time.clock()
print ("Total running time: ", end-start)