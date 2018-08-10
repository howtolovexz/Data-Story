import csv
import random
import operator
import pandas as pd
import numpy as np
from operator import itemgetter
from datetime import datetime
# dateList = ['2018-06-30', '2018-07-01', '2018-07-02', '2018-07-03', '2018-07-06', '2018-07-07', '2018-07-10',
#             '2018-07-11', '2018-07-14', '2018-07-15'] # match
dateList = ['2018-07-04', '2018-07-05', '2018-07-08', '2018-07-09',
            '2018-07-12', '2018-07-13'] # no match
inputFileList = ['../../data/NumbersTweets/worldcup' + date + 'nonnumbers.csv' for date in dateList]
outputFileName = '../../data/SampledData/nomatchNonNumbers500samples.csv'

# content = open('data/OriginalTweets/worldcup2018-07-03original.csv', "r", encoding='utf-8').read().replace('\r\n','\n')
#
# with open('data/OriginalTweets/worldcup2018-07-03originalx.csv', "w", encoding='utf-8') as g:
#     g.write(content)

colnames = ['created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status']
df = pd.DataFrame()

for inputFileName in inputFileList:
    # df_temp = pd.read_csv(inputFileName, names=colnames, encoding='utf-8', dtype={'followers': np.int, 'friends': np.int, 'rt': np.int, 'fav': np.int, 'retweeted_status': bool})
    df_temp = pd.read_csv(inputFileName, names=colnames, encoding='utf-8', lineterminator='\n', dtype={'text': str, 'followers': np.int, 'friends': np.int, 'rt': np.int, 'fav': np.int, 'retweeted_status': bool})
    df = df.append(df_temp)

df = df.sample(500)
df.to_csv(outputFileName, encoding='utf-8', index=False, header=False)


################################### OLD PART ##############################################
# date = "2018-06-30"
# inputFileName = "data/rawDataFromJSON/worldcup" + date + ".csv"
# outputFileName = "data/SampledData/worldcup" + date + "sampled.csv"
# # Open/Create a file to append data
# csvFile = open(inputFileName, 'r', encoding="utf-8")
# csvFile2 = open(outputFileName, 'w', encoding="utf-8")
# # Use csv Writer
# csvReader = csv.reader(csvFile, delimiter=',')
# csvWriter = csv.writer(csvFile2)
#
# count = 0
# text_list = []
# print("Start: " + str(datetime.now()))
# for row in csvReader:
#     text = ', '.join(row)
#     if text != "":
#         text_list.append(row)
#
# print("Finish: " + str(datetime.now()))
#
# random_choice = random.sample(text_list, 1000)
# random_choice2 = sorted(random_choice, key=itemgetter(0))
# csvWriter.writerows(random_choice2)
#
# csvFile.close()
# csvFile2.close()