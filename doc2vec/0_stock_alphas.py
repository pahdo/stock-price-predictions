"""
# Alpha by NLP: Augmenting stock data with Alpha values

Using a csv file with the schema:  
`   0      1      2      3      4      5      6      7      8      9      10`  
`Date   Symbol Return`  
We will create a csv file with the schema:  
`   0      1      2      3      4      5      6      7      8      9      10`  
`Date   Symbol Return Beta   Alpha`  

Running this notebook will take `/data/Stock_Returns/Original/stocks.csv` and augment each row with alpha values, i.e. the result of r<sub>stock</sub> - $\beta$r<sub>market</sub>, and create file `/data/Stock_Returns/Stocks_Plus_Alphas.csv`.
"""

# Configuration
dir_name = '/home/jovyan/text-analytics-for-accountancy/data/stocks'
stocks_csv = 'stocks.csv'
spy_csv = 'spy.csv'
output_file = 'stocks_alphas.csv'

import csv
import numpy as np
from datetime import datetime

dateformat = '%Y-%M-%d' # 1998-01-15

"""
Takes in an array rows containing the CSV rows for 1 stock symbol
Writes to output_file rows of the form
date symbol return beta alpha
"""
def write_csv_with_alphas(rows):
    with open(os.path.join(dir_name, spy_csv), 'rt') as SPY_File:
        SPY_Rows = []
        reader = csv.reader(SPY_File, delimiter=',')
        for row in reader:
            SPY_Rows.append(row)
        
        assert(len(rows) > 0)
        SPY_earliest_date_str = SPY_Rows[0][0]
        SPY_latest_date_str = SPY_Rows[-1][0]
        other_earliest_date_str = rows[0][0]
        other_latest_date_str = rows[-1][0]
        
        # strptime converts a string to a date
        SPY_earliest_date = datetime.strptime(SPY_earliest_date_str, dateformat) 
        SPY_latest_date = datetime.strptime(SPY_latest_date_str, dateformat)
        other_earliest_date = datetime.strptime(other_earliest_date_str, dateformat)
        other_latest_date = datetime.strptime(other_latest_date_str, dateformat)
        
        if (other_earliest_date > SPY_earliest_date):
            start_date = other_earliest_date
        else: # SPY_earliest_date >= other_earliest_date
            start_date = SPY_earliest_date
        if (other_latest_date < SPY_latest_date):
            end_date = other_latest_date
        else: # SPY_latest_date <= other_latest_date
            end_date = SPY_latest_date
            
        start_date_str = datetime.strftime(start_date, dateformat)
        end_date_str = datetime.strftime(end_date, dateformat)
        
        reached_start_date = False
        reached_end_date = False
            
        SPY_Returns = []
        SPY_Rows1 = []
        for row in SPY_Rows:
            if (row[0] == start_date_str):
                reached_start_date = True
            if (row[0] == end_date_str):
                reached_end_date = True
            if (reached_start_date and not reached_end_date):
                SPY_Returns.append(row[2])
                SPY_Rows1.append(row)
            
        reached_start_date = False
        reached_end_date = False
            
        Other_Returns = []
        Other_Rows = []
        for row in rows:
            if (row[0] == start_date_str):
                reached_start_date = True
            if (row[0] == end_date_str):
                reached_end_date = True
            if (reached_start_date and not reached_end_date):
                Other_Returns.append(row[2])
                Other_Rows.append(row)

        if (len(Other_Returns) != len(SPY_Returns)):
            if(len(Other_Rows)!=0):
                print("{} failed.".format(Other_Rows[0][1]))
            else:
                print("UNK failed")
            return

        SPY_Returns_arr = np.array(SPY_Returns, dtype=np.float32)
        Other_Returns_arr = np.array(Other_Returns, dtype=np.float32)
        # np.cov return value is [cov(a,a) cov(a,b)
        #                         cov(a,b) cov(b,b)]
        cov = np.cov(SPY_Returns_arr, Other_Returns_arr)
        beta = cov[0][1]/cov[0][0]

        
    with open(os.path.join(dir_name, output_file), 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        assert(len(SPY_Returns) == len(Other_Returns))   
        for i in range(len(SPY_Returns)):
            alpha = Other_Returns_arr[i] - beta * SPY_Returns_arr[i]
            writer.writerow(Other_Rows[i] + [beta, alpha])

"""
parse the stocks CSV and write to the output CSV every time we reach a new stock symbol
"""
import os.path
if (os.path.isfile(os.path.join(dir_name, output_file))):
    print("Please delete Stock_Plus_Alphas.csv before proceeding so we can regenerate the file.")
else:
    with open(os.path.join(dir_name, stocks_csv), 'rt') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        rows = []
        last_symbol = ''
        for row in reader:
            if(row[0] == 'date'):
                pass
            else:
                if (last_symbol != row[1]): # We have reached a new stock symbol
                    if (len(rows) >= 2):
                        write_csv_with_alphas(rows)
                    rows = []
                rows.append(row)
                last_symbol = row[1]

"""
verify our output file has been created
"""
cat_path = os.path.join(dir_name, output_file) 
bashCommand = 'cat {} | head -10'.format(cat_path)
import subprocess
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
print(output)