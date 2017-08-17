# Configuration
cikTickerPath = '../../GROUP_SHARED/data/10K/10-X_C/cikTicker.txt'
form10KPath = '../../GROUP_SHARED/data/10K/10-X_C/'

with open(cikTickerPath) as cikfile:
    cikTicker = cikfile.read()  
cikTicker = cikTicker.replace('\n', '')
cikTicker = cikTicker.replace('"', '')
cikTicker = cikTicker.replace('"', '')
cikTicker = cikTicker[2:-2].split('}, {')
cikdict = dict(s.split(', ') for s in cikTicker)

import sys
sys.path.append('../Database')
import getReturns
import numpy as np
import os
import re
import time
import sqlite3

start = time.clock()
conn = sqlite3.connect('../Database/stocks.db')
end = time.clock()
print("Connection open processor time: {0}".format(end-start))
results = getReturns.getRets(conn, 'AAPL', '2016-07-27')
print("AAPL Return on 2016-07-27: {0}".format(results[0][3]))

def parseTxtName(txt):
    txt = os.path.basename(txt)
    pattern = "edgar_data_(.*?)_"
    m = re.search(pattern, txt)
    if m:
        cik = m.group(1)
    pattern = "(\d{8})_"
    m = re.search(pattern, txt)
    if m:
        date = m.group(1)
    date = '{}-{}-{}'.format(date[0:4], date[4:6], date[6:])
    return cik, date
 
test_txt = '20160727_10-Q_edgar_data_320193_0001628280-16-017809_1.txt' # AAPL
print("Parsed title: {0}".format(parseTxtName(test_txt)))

def isPos(txt, cikdict): 
    #start = time.clock()
    cik, date = parseTxtName(txt)
    #end = time.clock()
    #print("Title parsing processor time: {0}".format(end-start))
    #start = time.clock()
    if cik in cikdict:
        ret = getReturns.getRets(conn, cikdict[cik], date, 4)
    else:
        raise Exception('Not in cikDict')
    #end = time.clock()
    #print("Query processor time: {0}".format(end-start))
    if (len(ret) != 0):
        return(np.sign(ret[0][3])==1.0)
    else:
        raise Exception('Query failed')
    
start = time.clock()
print("isPos = {0}".format(isPos(test_txt, cikdict)))
end = time.clock()
print("isPos processor time: {0}".format(end-start))

import shutil

folders = ['data/train/pos', 'data/train/neg', 'data/test/pos', 'data/test/neg', 'data/train/unsup']
for fol in folders:
    if not os.path.exists(fol):
        os.makedirs(fol)
    else:
        #shutil.rmtree(fol) # Clear old text files
        #os.makedirs(fol)
        print("Folders already made.")
        
import glob
import random

import sys

orig_stdout = sys.stdout
f = open('out_group_docs_by_sentiment.txt', 'w')
sys.stdout = f

#prefix = '/Users/daniel/Downloads/Versioned/text-analytics-for-accountancy/data/Form_10-Ks/'
#prefix = '../data/Form_10-Ks/'
quarters = ['2013/QTR2', '2013/QTR3', '2013/QTR4', 
            '2012/QTR1', '2012/QTR2', '2012/QTR3', '2012/QTR4', 
            '2011/QTR1', '2011/QTR2', '2011/QTR3', '2011/QTR4', 
            '2010/QTR1', '2010/QTR2', '2010/QTR3', '2010/QTR4', 
            '2009/QTR1', '2009/QTR2', '2009/QTR3', '2009/QTR4', 
            '2008/QTR1', '2008/QTR2', '2008/QTR3', '2008/QTR4', 
            '2007/QTR1', '2007/QTR2', '2007/QTR3', '2007/QTR4', 
            '2006/QTR1', '2006/QTR2', '2006/QTR3', '2006/QTR4', 
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
            '1994/QTR1', '1994/QTR2', '1994/QTR3', '1994/QTR4'
           ]
start_all = time.clock()
start_time_all = time.time()
for quarter in quarters:
    start = time.clock()
    start_time = time.time()
    print("Starting quarter {0}".format(quarter))
    dirname = os.path.join(form10KPath, quarter)
    txt_files = glob.glob(os.path.join(dirname, '*.txt'))
    for txt in txt_files:
        rand = random.random()
        #start = time.clock()
        try:
            pos = isPos(txt, cikdict)
        except Exception:
            continue
        #end = time.clock()
        #print("Query processor time: {0}".format(end-start))
        #start = time.clock()
        if (pos and rand <= 0.8):
            shutil.copy(txt, os.path.join(folders[0], os.path.basename(txt)))
        elif (pos and rand > 0.8): 
            shutil.copy(txt, os.path.join(folders[2], os.path.basename(txt)))
        elif (not pos and rand <= 0.8):
            shutil.copy(txt, os.path.join(folders[1], os.path.basename(txt)))
        else: # not pos and rand > 0.8
            shutil.copy(txt, os.path.join(folders[3], os.path.basename(txt)))
        #end = time.clock()
        #print("Copy processor time: {0}".format(end-start))
    end = time.clock()
    end_time = time.time()
    print("Processor time {0}".format(end-start))
    print("Running time {0}".format(end_time-start_time))
end_all = time.clock()
end_time_all = time.time()
print("Total processor time {0}".format(end_all-start_all))
print("Total running time {0}".format(end_time_all-start_time_all))

sys.stdout = orig_stdout
f.close()