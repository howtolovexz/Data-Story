import csv
import pandas
from datetime import datetime

date = "2018-06-30"
file_name = "data/OriginalTweets/worldcup" + date + "original.csv"
file_name2 = "data/test/worldcup" + date + "top.csv"

# # Open/Create a file to append data
# csvFile = open(file_name, 'r', encoding="utf-8")
# csvFile2 = open(file_name2, 'a', encoding="utf-8")
# # Use csv Writer
# csvReader = csv.reader(csvFile, delimiter=',')
# csvWriter = csv.writer(csvFile2)
#
# count = 0
# text_list = []
# print("Start: " + str(datetime.now()))
# for row in csvReader:
#     text = ', '.join(row)
#     if text != "" and "rt @" not in text:
#         print(row)
#         # csvWriter.writerow(row)
#         # text_list.append(text)
#         # print(text)
#
# csvFile.close()
# csvFile2.close()
# print("Finish: " + str(datetime.now()))

colnames = ['date', 'text', 'retweet', 'platform']
data = pandas.read_csv(file_name, names=colnames)
data = data.sort_values(by=['retweet'], ascending=False)
# data.to_csv(file_name2, index=False, encoding='utf-8')