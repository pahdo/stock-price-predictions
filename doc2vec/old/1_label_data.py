"""
1_label_data.py labels Form 10-Ks by sentiment
"""

from os import path
import os
import numpy as np
import re
import time
import sqlite3
import datetime
import shutil
import glob
import random
from sys import stdout

###### CONFIGURATION ######
path_to_data = '../data/'
db_path = path.join(path_to_data, 'database/stocks.db')
sec_path = path.join(path_to_data, '10-X_C/')
cik_ticker_path = path.join(sec_path, 'cikTicker.txt')

"""
mode:
    mode = 0 Label documents by one-day alphas on the Form 10-K release date
    mode = 1 Label documents by thirty-day cumulative returns from Form 10-K release date
    mode = 2 Label documents by cumulative alpha in an 9-day window around the Form 10-K release date
"""
mode = 2

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
###########################

def check_file(filename):
    """Check presence of a file.

    Returns:
        bool: True for file exists and has size greater than 0. False otherwise.
    """
    try:
        if os.stat(filename).st_size > 0:
            return True
        else:
            return False
    except OSError:
        return False

def build_cik_dict():
    """Parses cikTicker.txt file into a dictionary.

    Returns:
        Dictionary object if cikTicker.txt is valid. None otherwise.
    """
    if check_file(cik_ticker_path):
        with open(cik_ticker_path) as cik_file:
            cik_ticker = cik_file.read()  
        cik_ticker = cik_ticker.replace('\n', '')
        cik_ticker = cik_ticker.replace('"', '')
        cik_ticker = cik_ticker.replace('"', '')
        cik_ticker = cik_ticker[2:-2].split('}, {')
        cik_dict = dict(s.split(', ') for s in cik_ticker)
        return cik_dict
    else:
        print('cikTicker.txt does not exist.')
        return None

def get_rets(conn, symbol, date, horizon=1):
    c = conn.cursor()
    
    results = []
    for row in c.execute("SELECT theDate, Symbol, Return, Alpha \
                         FROM stocks WHERE symbol=? \
                         AND theDate >= strftime(?) \
                         AND theDate < date(strftime(?), ?);", 
                         [symbol, date, date, '+{} day'.format(horizon)]):
        results.append(row)
    return results

def date_subtract(date_str, delta):
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d') - datetime.timedelta(days=delta)
    return date.strftime('%Y-%m-%d')

def get_total_ret(conn, symbol, date, horizon=1, subtract=False):
    rets = get_rets(conn, symbol, date, horizon)
    the_rets = []
    for ret in rets:
        the_rets.append(ret[2])
    return np.product([1 + float(the_ret) for the_ret in the_rets])-1

def get_total_ret_trading_days(conn, symbol, theDate, window):
    rets = get_rets(conn, symbol, date_subtract(theDate, 4), horizon=window+5)
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

def parse_txt_name(txt):
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

def is_pos(conn, txt, cik_dict, mode): 
    cik, date = parse_txt_name(txt)
    if cik in cik_dict:
        if mode == 0:
            ret = get_rets(conn, cik_dict[cik], date, 4) 
        elif mode == 1:
            ret = get_total_ret(conn, cik_dict[cik], date, 30)
        elif mode == 2:
            stock_ret = get_total_ret_trading_days(conn, cik_dict[cik], date_subtract(date, 4), 9)
            spy_ret = get_total_ret_trading_days(conn, 'SPY', date_subtract(date, 4), 9)
            beta = get_rets(conn, cik_dict[cik], date)[0][3]
            ret = stock_ret - spy_ret * beta
    else:
        return 0
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

def checks_pass(conn, cik_dict):
    """Test the database connection and file paths for validity.

    Args:
        sqlite3 connection object.
    Returns:
        bool: True if valid. False otherwise.
    """
    # TODO: Add actual checks.
    results = get_rets(conn, 'AAPL', '2016-07-27')
    print("AAPL Return on 2016-07-27: {0}".format(results[0][3]))

    test_txt = '20160727_10-Q_edgar_data_320193_0001628280-16-017809_1.txt' # AAPL
    print("Parsed title: {0}".format(parse_txt_name(test_txt)))

    start = time.clock()
    print("is_pos = {0}".format(is_pos(conn, test_txt, cik_dict, mode)))
    end = time.clock()
    print("is_pos processor time: {0}".format(end-start))
    return True
    
def main():
    conn = sqlite3.connect(db_path)
    cik_dict = build_cik_dict()
    if not checks_pass(conn, cik_dict):
        print("Checks fail.")
        return

    """ Copies and groups documents into folders by sentiment """
    if mode == 0:
        print("Unsupported mode.")
        return
        folders = ['data/pos', 'data/neg']
    elif mode == 1:
        print("Unsupported mode.")
        return
        folders = ['data_by_returns/pos', 'data_by_returns/neg']
    elif mode == 2:
        prefix = 'data_by_cum_alphas'
        labels = ['neg', 'unsupervised', 'neutral', 'pos']

    for fol in folders:
        if not os.path.exists(fol):
            os.makedirs(fol)
        else:
            print("Folder {} already created".format(fol))

    start_clock_total = time.clock()     # Processor time
    start_time_total = time.time()       # Clock time

    quarters = [train_quarters, validation_quarters, test_quarters]
    for i in range(len(quarters)):
        for quarter in quarters[i]:
            start_clock = time.clock()
            start_time = time.time()
            print("Starting quarter {0}.".format(quarter))
            dirname = path.join(sec_path, quarter)
            txt_files = glob.glob(path.join(dirname, '*.txt'))
            for txt in txt_files:
                pos = is_pos(conn, txt, cik_dict, mode)
                if mode == 2:
                    if i == 0:
                        split = 'train'
                    elif i == 1:
                        split = 'validation'
                    elif i == 2:
                        split = 'test'
                    if pos == -1:
                        shutil.copy(txt, path.join(prefix, split, labels[0])) # neg
                    elif pos == 0:
                        shutil.copy(txt, path.join(prefix, split, labels[1])) # unsupervised
                    elif pos == 1:
                        shutil.copy(txt, path.join(prefix, split, labels[2])) # neutral
                    elif pos == 2:
                        shutil.copy(txt, path.join(prefix, split, labels[3])) # pos
            end_clock = time.clock()
            end_time = time.time()
            print("Processor time {0}.".format(end_clock-start_clock))
            print("Running time {0}.".format(end_time-start_time))
            stdout.flush()

    end_clock_total = time.clock()
    end_time_total = time.time()
    print("Total processor time {0}.".format(end_clock_total-start_clock_total))
    print("Total running time {0}.".format(end_time_total-start_time_total))

if __name__ == "__main__":
    main()