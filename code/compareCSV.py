import csv

inputFileName1 = '../../data/SampledData/ver 2/traintest.csv'
inputFileName2 = '../../data/SampledData/ver 2/matchlabelled.csv'
outputFileName = '../../data/SampledData/ver 2/temp.csv'
inputFile1 = open(inputFileName1, 'r', encoding="utf-8")
inputFile2 = open(inputFileName2, 'r', encoding="utf-8")
outputFile = open(outputFileName, 'w', encoding="utf-8")