import os

files = ["2017_data.json", "2018_data.json", "2019_data.json"]

with open('../data/full_data2.json', 'w+') as outfile:
    for f in files:
        with open('../data/'+f, 'r') as infile:
            outfile.writelines(infile.readlines())