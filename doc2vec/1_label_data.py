"""
1_label_data.py labels Form 10-Ks by sentiment
"""

import os.path
###### CONFIGURATION ######
path_to_data = '../data/'
db_path = os.path.join(path_to_data, 'database/stocks.db')
sec_path = os.path.join(path_to_data, '10-X_C/')
cik_ticker_path = os.path.join(sec_path, 'cikTicker.txt')

"""
mode:
    mode = 0 Label documents by one-day alphas on the Form 10-K release date
    mode = 1 Label documents by thirty-day cumulative returns from Form 10-K release date
    mode = 2 Label documents by cumulative alpha in an 9-day window around the Form 10-K release date
"""
mode = 2
###########################

"""
Builds cikdict: map of cik values to stock tickers
"""
with open(cik_ticker_path) as cik_file:
    cik_ticker = cik_file.read()  
cik_ticker = cik_ticker.replace('\n', '')
cik_ticker = cik_ticker.replace('"', '')
cik_ticker = cik_ticker.replace('"', '')
cik_ticker = cik_ticker[2:-2].split('}, {')
cik_dict = dict(s.split(', ') for s in cik_ticker)

import numpy as np
import re
import time
import sqlite3
import datetime

def getRets(conn, symbol, date, horizon=1):
    c = conn.cursor()
    
    results = []
    for row in c.execute("SELECT theDate, Symbol, Return, Alpha \
                         FROM stocks WHERE symbol=? \
                         AND theDate >= strftime(?) \
                         AND theDate < date(strftime(?), ?);", 
                         [symbol, date, date, '+{} day'.format(horizon)]):
        results.append(row)
    return results

def dateSubtract(date_str, delta):
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d') - datetime.timedelta(days=delta)
    return date.strftime('%Y-%m-%d')

def getTotalRet(conn, symbol, date, horizon=1, subtract=False):
    rets = getRets(conn, symbol, date, horizon)
    the_rets = []
    for ret in rets:
        the_rets.append(ret[2])
    return np.product([1 + float(the_ret) for the_ret in the_rets])-1

def getTotalRetTradingDays(conn, symbol, theDate, window):
    rets = getRets(conn, symbol, dateSubtract(theDate, 4), horizon=window+5)
    for i in range(len(rets)):
        if (datetime.datetime.strptime(rets[i][0], '%Y-%m-%d') == datetime.datetime.strptime(theDate, '%Y-%m-%d')):
            middle_idx = i  
        else:
            print("{} vs {}".format(datetime.datetime.strptime(rets[i][0], '%Y-%m-%d'), datetime.datetime.strptime(theDate, '%Y-%m-%d')))
    rets = rets[middle_idx-window:middle_idx+window+1]
    the_rets = []
    for ret in rets:
        the_rets.append(ret[2])
    return np.product([1 + float(the_ret) for the_ret in the_rets])-1

conn = sqlite3.connect(db_path)
results = getRets(conn, 'AAPL', '2016-07-27')
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

def isPos(txt, cik_dict, mode): 
    cik, date = parseTxtName(txt)
    if cik in cik_dict:
        if mode == 0:
            ret = getRets(conn, cik_dict[cik], date, 4) 
        elif mode == 1:
            ret = getTotalRet(conn, cik_dict[cik], date, 30)
        elif mode == 2:
            stock_ret = getTotalRetTradingDays(conn, cik_dict[cik], dateSubtract(date, 4), 9)
            spy_ret = getTotalRetTradingDays(conn, 'SPY', dateSubtract(date, 4), 9)
            beta = getRets(conn, cik_dict[cik], date)[0][3]
            ret = stock_ret - spy_ret * beta
    else:
        return 0
        #raise Exception('Not in cik_dict')
    if mode == 0:
        if len(ret) != 0: 
            return np.sign(ret[0][3])==1.0
        else:
            raise Exception('Query failed')
    elif mode == 1:
        return np.sign(ret)==1.0
    elif mode == 2:
        if ret < -0.01:
            return -1
        elif ret >= -0.01 and ret < 0.01:
            return 1
        else: # ret >= 0.01:
            return 2
    
start = time.clock()
print("isPos = {0}".format(isPos(test_txt, cik_dict, mode)))
end = time.clock()
print("isPos processor time: {0}".format(end-start))

import shutil

""" Copies and groups documents into folders by sentiment """
if mode == 0:
    folders = ['data/pos', 'data/neg']
elif mode == 1:
    folders = ['data_by_returns/pos', 'data_by_returns/neg']
elif mode == 2:
    folders = ['data_by_cum_alphas/pos', 'data_by_cum_alphas/neutral', 'data_by_cum_alphas/neg', 'data_by_cum_alphas/unsupervised']

for fol in folders:
    if not os.path.exists(fol):
        os.makedirs(fol)
    else:
        print("Folder {} already created".format(fol))
        
import glob
import random

train_quarters = [
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
validation_quarters = [ 
    '2009/QTR1', '2009/QTR2', '2009/QTR3', '2009/QTR4', 
    '2008/QTR1', '2008/QTR2', '2008/QTR3', '2008/QTR4',
    '2007/QTR1', '2007/QTR2', '2007/QTR3', '2007/QTR4', 
    '2006/QTR1', '2006/QTR2', '2006/QTR3', '2006/QTR4']
test_quarters = [
    '2013/QTR2', '2013/QTR3', '2013/QTR4', 
    '2012/QTR1', '2012/QTR2', '2012/QTR3', '2012/QTR4', 
    '2011/QTR1', '2011/QTR2', '2011/QTR3', '2011/QTR4',
    '2010/QTR1', '2010/QTR2', '2010/QTR3', '2010/QTR4']

start_clock_total = time.clock()     # Processor time
start_time_total = time.time()       # Clock time

quarters = [train_quarters, validation_quarters, test_quarters]
for i in range(len(quarters)):
    for quarter in quarters[i]:
        start_clock = time.clock()
        start_time = time.time()
        print("Starting quarter {0}".format(quarter))
        dirname = os.path.join(sec_path, quarter)
        txt_files = glob.glob(os.path.join(dirname, '*.txt'))
        for txt in txt_files:
    #        try:
            pos = isPos(txt, cik_dict, mode)
    #        except Exception:
    #            continue
            if mode != 2:
                print("unsupported")
                break
            elif mode == 2:
                if i == 0:
                    split = 'train'
                elif i == 1:
                    split = 'validation'
                elif i == 2:
                    split = 'test'
                if pos == -1:
                    shutil.copy(txt, os.path.join(split, folders[2])) # neg
                elif pos == 0:
                    shutil.copy(txt, os.path.join(split, folders[3])) # unsupervised
                elif pos == 1:
                    shutil.copy(txt, os.path.join(split, folders[1])) # neutral
                elif pos == 2:
                    shutil.copy(txt, os.path.join(split, folders[0])) # pos
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