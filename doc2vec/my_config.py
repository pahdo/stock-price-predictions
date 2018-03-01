###### CONFIGURATION - prep_data_v2.py ######

# train_quarters = [
#     '2017/QTR1']
#train_quarters = [
#    '2017/QTR2']
train_quarters = [
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
"""
train_quarters1 = [
     '2009/QTR1', '2009/QTR2', '2009/QTR3', '2009/QTR4',
]
train_quarters2 = ['2008/QTR1', '2008/QTR2', '2008/QTR3', '2008/QTR4',
     '2007/QTR1', '2007/QTR2', '2007/QTR3', '2007/QTR4',
]
train_quarters3 = [
     '2006/QTR1', '2006/QTR2', '2006/QTR3', '2006/QTR4',
     '2005/QTR1', '2005/QTR2', '2005/QTR3', '2005/QTR4',
]
train_quarters4 = [
     '2004/QTR1', '2004/QTR2', '2004/QTR3', '2004/QTR4',
     '2003/QTR1', '2003/QTR2', '2003/QTR3', '2003/QTR4',
]
train_quarters5 = [
     '2002/QTR1', '2002/QTR2', '2002/QTR3', '2002/QTR4',
     '2001/QTR1', '2001/QTR2', '2001/QTR3', '2001/QTR4',
]
train_quarters6 = [
     '2000/QTR1', '2000/QTR2', '2000/QTR3', '2000/QTR4',
     '1999/QTR1', '1999/QTR2', '1999/QTR3', '1999/QTR4',
     '1998/QTR1', '1998/QTR2', '1998/QTR3', '1998/QTR4',
]
train_quarters7 = [
     '1997/QTR1', '1997/QTR2', '1997/QTR3', '1997/QTR4',
     '1996/QTR1', '1996/QTR2', '1996/QTR3', '1996/QTR4',
     '1995/QTR1', '1995/QTR2', '1995/QTR3', '1995/QTR4',
     '1994/QTR1', '1994/QTR2', '1994/QTR3', '1994/QTR4']
"""
source_dir = '10-X_C'
output_dir = '10-X_C_clean'
dataset_dir = 'dataset'

###########################

###### CONFIGURATION - tf_idf_v2.py######

#test_quarters = []
test_quarters = [
    '2013/QTR2', '2013/QTR3', '2013/QTR4',
    '2012/QTR1', '2012/QTR2', '2012/QTR3', '2012/QTR4',
    '2011/QTR1', '2011/QTR2', '2011/QTR3', '2011/QTR4',
    '2010/QTR1', '2010/QTR2', '2010/QTR3', '2010/QTR4']

data_dir = '10-X_C_clean'
cache_dir = 'tmp'  # cache directory for pipeline

###########################
