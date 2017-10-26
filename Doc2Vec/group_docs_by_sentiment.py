"""
group_docs_by_sentiment.py groups documents by sentiment but no longer splits them into test/train/validation
"""

###### CONFIGURATION ######
cikTickerPath = '../../GROUP_SHARED/data/10K/10-X_C/cikTicker.txt'
form10KPath = '../../GROUP_SHARED/data/10K/10-X_C/'

"""
alpha = True:  Label documents by one day alphas on the date of SEC Form release
alpha = False: Label documents by thirty day cumulative returns from SEC Form release
"""
alpha = False
###########################

# Prepares cikdict, which is a map of cik values to stock tickers
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

conn = sqlite3.connect('../Database/stocks.db')
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

def isPos(txt, cikdict, alpha): 
    cik, date = parseTxtName(txt)
    if cik in cikdict:
        if (alpha):
            ret = getReturns.getRets(conn, cikdict[cik], date, 4) 
        else: # 30 day cumulative returns
            ret = getReturns.getTotalRet(conn, cikdict[cik], date, 30)
    else:
        raise Exception('Not in cikDict')
    if (alpha):
        if (len(ret) != 0): 
            return(np.sign(ret[0][3])==1.0)
        else:
            raise Exception('Query failed')
    else: # 30 day cumulative returns
        return(np.sign(ret)==1.0)
    
start = time.clock()
print("isPos = {0}".format(isPos(test_txt, cikdict, alpha)))
end = time.clock()
print("isPos processor time: {0}".format(end-start))

import shutil

"""
# Old behavior: this script would copy and split documents into train-pos/train-neg/test-pos/test-neg
if (alpha):
    folders = ['data/train/pos', 'data/train/neg', 'data/test/pos', 'data/test/neg', 'data/train/unsup']
else: # 30 day returns
    folders = ['data_by_returns/train/pos', 'data_by_returns/train/neg', 'data_by_returns/test/pos', 'data_by_returns/test/neg', 'data_by_returns/train/unsup']
"""
# New behavior: this script copies and groups documents by sentiment only so they can be easily labeled later
if (alpha):
    folders = ['data/pos', 'data/neg']
else:
    folders = ['data_by_returns/pos', 'data_by_returns/neg']

for fol in folders:
    if not os.path.exists(fol):
        os.makedirs(fol)
    else:
        print("Folder {} already created".format(fol))
        
import glob
import random

# Redirect output to group_docs_by_sentiment.out
orig_stdout = sys.stdout
f = open('group_docs_by_sentiment.out', 'w')
sys.stdout = f

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
            '1994/QTR1', '1994/QTR2', '1994/QTR3', '1994/QTR4']

start_clock_total = time.clock()     # Processor time
start_time_total = time.time()       # Time in seconds

for quarter in quarters:
    start_clock = time.clock()
    start_time = time.time()
    print("Starting quarter {0}".format(quarter))
    dirname = os.path.join(form10KPath, quarter)
    txt_files = glob.glob(os.path.join(dirname, '*.txt'))
    for txt in txt_files:
        """
        # Old behavior: this script would copy and split documents into train-pos/train-neg/test-pos/test-neg
        rand = random.random()
        """
        try:
            pos = isPos(txt, cikdict, alpha)
        except Exception:
            continue
        """
        # Old behavior: this script would copy and split documents into train-pos/train-neg/test-pos/test-neg
        if (pos and rand <= 0.8):
            shutil.copy(txt, os.path.join(folders[0], os.path.basename(txt)))
        elif (pos and rand > 0.8): 
            shutil.copy(txt, os.path.join(folders[2], os.path.basename(txt)))
        elif (not pos and rand <= 0.8):
            shutil.copy(txt, os.path.join(folders[1], os.path.basename(txt)))
        else: # not pos and rand > 0.8
            shutil.copy(txt, os.path.join(folders[3], os.path.basename(txt)))
        """
        # New behavior: this script copies and groups documents by sentiment only so they can be easily labeled later
        if pos:
            shutil.copy(txt, os.path.join(folders[0]))
        else:
            shutil.copy(txt, os.path.join(folders[1]))
    end_clock = time.clock()
    end_time = time.time()
    print("Processor time {0}".format(end_clock-start_clock))
    print("Running time {0}".format(end_time-start_time))
    sys.stdout.flush()

end_clock_total = time.clock()
end_time_total = time.time()
print("Total processor time {0}".format(end_clock_total-start_clock_total))
print("Total running time {0}".format(end_time_total-start_time_total))

sys.stdout = orig_stdout
f.close()
