#!/usr/bin/env python3
import fileinput

date = "2018-06-30"
filename = "data/rawData/worldcup" + date + "-7.txt"
filename2 = "data/rawData/worldcup" + date + "-7x.txt"

f1 = open(filename, 'r', encoding='utf-8')
f2 = open(filename2, 'w', encoding='utf-8')
for line in f1:
    f2.write(line.replace('}{', '},{'))
f1.close()
f2.close()