import datetime
import glob
import os
import re
import numpy as np
import sqlite3

###### CONFIGURATION ######

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
    '1994/QTR1', '1994/QTR2', '1994/QTR3', '1994/QTR4',
    '2009/QTR1', '2009/QTR2', '2009/QTR3', '2009/QTR4',
    '2008/QTR1', '2008/QTR2', '2008/QTR3', '2008/QTR4',
    '2007/QTR1', '2007/QTR2', '2007/QTR3', '2007/QTR4',
    '2006/QTR1', '2006/QTR2', '2006/QTR3', '2006/QTR4']
test_quarters = [
    '2013/QTR2', '2013/QTR3', '2013/QTR4',
    '2012/QTR1', '2012/QTR2', '2012/QTR3', '2012/QTR4',
    '2011/QTR1', '2011/QTR2', '2011/QTR3', '2011/QTR4',
    '2010/QTR1', '2010/QTR2', '2010/QTR3', '2010/QTR4']
cik_ticker_path = os.path.join('..', 'data', 'cikTicker.txt')
db_path = os.path.join('..', 'data', 'database', 'stocks.db')

###########################

def load_texts(directory, split=['all', 'train', 'test'], yield_paths=False, yield_labels=False):
    """assumes source directory structure: ../data/10-X_C/2004/QTR2/
    """
    if split == 'all':
        regex_part = '**'
    elif split == 'train':
        """https://stackoverflow.com/questions/33406313/how-to-match-any-string-from-a-list-of-strings-in-regular-expressions-in-python
        """
        regex_part = '|'.join(train_quarters)
    elif split == 'test':
        regex_part = '|'.join(test_quarters)
    regex = os.path.join('../data', directory, regex_part, '*.txt')
    print(regex)
    file_paths = glob.iglob(regex, recursive=True)
    if yield_paths:
        for file_path in file_paths:
            with open(file_path, 'r') as t:
                yield t.read()
    else:
        for file_path in file_paths:
            with open(file_path, 'r') as t:
                yield t.read(), file_path # file_path is a full path

def load_labels(directory, split=['all', 'train', 'test']):
    if split == 'all':
        regex_part = '**'
    elif split == 'train':
        regex_part = '|'.join(train_quarters)
    elif split == 'test':
        regex_part = '|'.join(test_quarters)
    regex = os.path.join('..', 'data', directory, regex_part, '*.txt')
    print(regex)
    file_paths = glob.iglob(regex, recursive=True)
    #print(list(file_paths))
    conn = sqlite3.connect(db_path)
    cik_dict = build_cik_dict()
    #if cik_dict is None:
    #    return
    for txt in file_paths:
        print(txt)
        cik, date = parse_txt_name(txt)
        print("{} {}".format(cik, date))
        if cik in cik_dict:
            price_history, alpha1, alpha2, alpha3, alpha4, alpha5 = get_returns(conn, cik_dict[cik], date)
            print("{} {} {} {} {} {}".format(price_history, alpha1, alpha2, alpha3, alpha4, alpha5))

def get_returns(conn, symbol, date, horizon=12, normalized=True):
    # date_subtract: 3 day buffer for weekends + 5 day for baseline price history features
    # horizon: 3 day buffer for weekends + 8 from date subtract + 1 (because upper bound is exclusive)
    if normalized:
        c = conn.cursor()
        start_date = date_subtract(date, 8)
        stock_returns = []
        for row in c.execute("SELECT theDate, Symbol, Return, Alpha \
                             FROM stocks WHERE symbol=? \
                             AND theDate >= strftime(?) \
                             AND theDate < date(strftime(?), ?);", 
                             [symbol, start_date, start_date, '+{} day'.format(horizon)]):
            stock_returns.append(row)
        spy_returns = []
        for row in c.execute("SELECT theDate, Symbol, Return, Alpha \
                             FROM stocks WHERE symbol=? \
                             AND theDate >= strftime(?) \
                             AND theDate < date(strftime(?), ?);",
                             ['SPY', start_date, start_date, '+{} day'.format(horizon)]):
            spy_returns.append(row)
        if len(stock_returns != len(spy_returns)):
            raise Exception()# invalid because missing value
        for i in range(len(stock_returns)):
            if datetime.datetime.strptime(stock_returns[i][0], '%Y-%m-%d') != datetime.datetime.strptime(spy_returns[i][0], '%Y-%m-%d'):
                raise Exception()# invalid because non-matching dates
        for i in range(len(stock_returns)):
            if datetime.datetime.strptime(stock_returns[i][0], '%Y-%m-%d') >= datetime.datetime.strptime(date, '%Y-%m-%d'):
                middle_idx = i # search for "middle_idx" or day 1, the day the Form 10-K is filed
                break
        if middle_idx < 5:
            raise Exception() # invalid because not enough price history
        if len(stock_returns) - middle_idx < 5:
            raise Exception() # invalid because not enough future price data
        beta = float(stock_returns[0][3])
        stock_returns = [float(ret[2]) for ret in stock_returns]
        spy_returns = [float(ret[2]) for ret in stock_returns]
        price_history = np.subtract(stock_returns[middle_idx-5, middle_idx], spy_returns[middle_idx-5, middle_idx])
        alpha1 = np.product([1 + ret for ret in stock_returns[middle_idx, middle_idx+1]]) - beta * np.product([1 + ret for ret in spy_returns[middle_idx, middle_idx+1]])
        alpha2 = np.product([1 + ret for ret in stock_returns[middle_idx, middle_idx+2]]) - beta * np.product([1 + ret for ret in spy_returns[middle_idx, middle_idx+2]])
        alpha3 = np.product([1 + ret for ret in stock_returns[middle_idx, middle_idx+3]]) - beta * np.product([1 + ret for ret in spy_returns[middle_idx, middle_idx+3]])
        alpha4 = np.product([1 + ret for ret in stock_returns[middle_idx, middle_idx+4]]) - beta * np.product([1 + ret for ret in spy_returns[middle_idx, middle_idx+4]])
        alpha5 = np.product([1 + ret for ret in stock_returns[middle_idx, middle_idx+5]]) - beta * np.product([1 + ret for ret in spy_returns[middle_idx, middle_idx+5]])
        return (price_history, alpha1, alpha2, alpha3, alpha4, alpha5)

    else:
        print("Not supported.")

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

def date_subtract(date_str, delta):
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d') - datetime.timedelta(days=delta)
    return date.strftime('%Y-%m-%d')

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