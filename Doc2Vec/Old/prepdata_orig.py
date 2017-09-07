### Original Prep Data - Fast, but doesn't stem, remove OOV, or remove stop words

locale.setlocale(locale.LC_ALL, 'C')
if sys.version > '3':
    control_chars = [chr(0x85)]
else:
    control_chars = [unichr(0x85)]

# Convert text to lower-case and strip punctuation and symbols from words
def normalize_text(text):
    norm_text = text.lower()
    norm_text = norm_text.replace('<br />', ' ')
    norm_text = norm_text.replace('\r', ' ')
    norm_text = norm_text.replace('\n', ' ')
    #Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':', '{', '}', '<', '>']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    re.sub('\s+', ' ', norm_text).strip()
    return norm_text

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
                t_clean = t.read().decode("utf-8")
                for c in control_chars:
                    t_clean = t_clean.replace(c, ' ')                
                t_clean = normalize_text(t_clean)
                temp += t_clean
            temp += "\n"
        output = fol.replace('/', '-') + '.txt'
        print("{} aggregated".format(os.path.join(dirname, output)))
        with smart_open.smart_open(os.path.join(dirname, output), "wb") as n:
            for idx, line in enumerate(temp.split('\n')):
                num_line = u"_*{0} {1}\n".format(idx, line)
                n.write(num_line.encode("UTF-8"))
        alldata += temp
    with smart_open.smart_open(os.path.join(dirname, 'alldata-id.txt'), 'wb') as f:
        for idx, line in enumerate(alldata.split('\n')):
            num_line = u"_*{0} {1}\n".format(idx, line)
            f.write(num_line.encode("UTF-8")) 
    print("alldata-id.txt aggregated")
    return alldata

start = time.clock()
alldata = prepData()
end = time.clock()
print ("Total running time: ", end-start)