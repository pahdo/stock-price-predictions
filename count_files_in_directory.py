"""paths = ['Doc2Vec/data_by_returns_small/pos',
         'Doc2Vec/data_by_returns_small/neg',
         'Doc2Vec/data_by_returns/pos',
         'Doc2Vec/data_by_returns/neg',
         'Doc2Vec/data/pos',
         'Doc2Vec/data/neg',
         'Doc2Vec/data_small/pos',
         'Doc2Vec/data_small/neg',
         'Doc2Vec/data_by_returns_small/train/pos',
         'Doc2Vec/data_by_returns_small/train/neg',
         'Doc2Vec/data_by_returns_small/test/pos',
         'Doc2Vec/data_by_returns_small/test/neg',
         'Doc2Vec/data_small/train/pos',
         'Doc2Vec/data_small/train/neg',
         'Doc2Vec/data_small/test/pos',
         'Doc2Vec/data_small/test/neg',
         'Doc2Vec/data_small/valid/pos',
         'Doc2Vec/data_small/valid/neg']"""
"""paths = ['Doc2Vec/data_by_returns/pos',
            'Doc2Vec/data_by_returns/neg']"""
form10KPath = '../GROUP_SHARED/data/10K/10-X_C/'
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
import os.path
paths = [os.path.join(form10KPath, quarter) for quarter in quarters]
print(paths)
for path in paths:
    num_files = len([f for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))])
    print("Path={} num_files={}".format(path, num_files))
