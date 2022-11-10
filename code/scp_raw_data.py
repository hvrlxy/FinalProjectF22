import os, sys 

for id in range(4, 107):
    # make a directory of named DS_id in the data directory
    os.mkdir('data/raw/DS_' + str(id))
    # copy the data from the server to the directory