import csv
from datetime import datetime

date = "2018-06-29"
file_name1 = "data/rawData/worldcup" + date + ".csv"
file_name2 = "data/rawData/worldcup" + date + "-2.csv"
file_name3 = "data/rawData/worldcup" + date + "-3.csv"
# file_name4 = "data/rawData/worldcup" + date + "-4.csv"
# file_name5 = "data/rawData/worldcup" + date + "-5.csv"
# file_name6 = "data/rawData/worldcup" + date + "-6.csv"
file_name_original = "data/OriginalTweets/worldcup" + date + "original.csv"
file_name_list = [file_name1, file_name2, file_name3]
# Open/Create a file to append data
# csvFile = open(file_name, 'r', encoding="utf-8")
csvFile2 = open(file_name_original, 'a', encoding="utf-8")
# Use csv Writer
# csvReader = csv.reader(csvFile, delimiter=',')
csvWriter = csv.writer(csvFile2)

count = 0
text_list = []
datetime_temp = "23:59:59"
print("Start: " + str(datetime.now()))
for file_name_temp in file_name_list:
    csvFile = open(file_name_temp, 'r', encoding="utf-8")
    csvReader = csv.reader(csvFile, delimiter=',')
    for row in csvReader:
        text = ', '.join(row)
        if text != "" and "rt @" not in text:
            if row[0] < datetime_temp:
                print(row)
                datetime_temp = row[0]
                csvWriter.writerow(row)
            # text_list.append(text)
            # print(text)
    csvFile.close()

csvFile.close()
csvFile2.close()
print("Finish: " + str(datetime.now()))