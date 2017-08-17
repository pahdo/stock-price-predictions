#import sqlite3
#import time

def getRets(conn, symbol, date, horizon=1):
    # conn = sqlite3.connect('../Database/stocks.db')
    #start = time.clock()
    c = conn.cursor()
    #end = time.clock()
    #print("Cursor processor time: {0}".format(end-start))
    
    results = []
    #start = time.clock()
    for row in c.execute("SELECT theDate, Symbol, Return, Alpha \
                         FROM stocks WHERE symbol=? \
                         AND theDate >= strftime(?) \
                         AND theDate < date(strftime(?), ?);", 
                         [symbol, date, date, '+{} day'.format(horizon)]):
        results.append(row)
    #end = time.clock()
    #print("Query execution processor time: {0}".format(end-start))
    return results