paths = ['Doc2Vec/data_by_returns_small/pos',
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
         'Doc2Vec/data_small/valid/neg']
import os.path
import subprocess
import sys
for path in paths:
    bashCommand = "ls {} | head -20".format(path)
    print("path={}".format(path))
    sys.stdout.flush()
    process = subprocess.Popen(bashCommand, shell=True)
    process.wait()
