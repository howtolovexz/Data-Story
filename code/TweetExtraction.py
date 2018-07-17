import csv
import json
from datetime import datetime

def txtToJSON(inputFileList):
    for inputFile in inputFileList:
        outputFile = inputFile.replace('.txt', 'new.txt')
        outputFile = outputFile.replace('C:/Users/JaZz-/Documents/Dissertation/data/', 'data/rawData/')
        print('Working on file: ' + inputFile)
        txtInput = open(inputFile, 'r', encoding='utf-8')
        txtOutput = open(outputFile, 'w', encoding='utf-8')
        txtOutput.write('[')

        for line in txtInput:
            txtOutput.write(line.replace('}{', '},{'))

        txtOutput.write(']')
        txtInput.close()
        txtOutput.close()

# def originalTweetExtraction(inputFileList, outputFlie):
#     csvOutput = open(outputFlie, 'a', encoding="utf-8")
#     csvWriter = csv.writer(csvOutput)
#
#     datetime_temp = "23:59:59"
#     for inputFile in inputFileList:
#         csvInput = open(inputFile, 'r', encoding="utf-8")
#         csvReader = csv.reader(csvInput, delimiter=',')
#         for row in csvReader:
#             text = ', '.join(row)
#             if text != "" and "rt @" not in text: #exclude retweet and null rows
#                 if row[0] < datetime_temp: #remove duplicate tweet
#                     datetime_temp = row[0]
#                     csvWriter.writerow(row)
#         csvInput.close()
#
#     csvOutput.close()

def JSONToCsvExtraction(inputFileList, outputFile):
    csvOutput = open(outputFile, mode='w', encoding='utf-8')  # opens csv file
    writer = csv.writer(csvOutput)  # create the csv writer object

    # fields = ['created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status']  # field names
    # writer.writerow(fields)  # writes field

    for inputFile in inputFileList:
        print('Working on file: ' + inputFile)
        data_json = open(inputFile, mode='r', encoding='utf-8').read()  # reads in the JSON file into Python as a string
        data_python = json.loads(data_json)  # turns the string into a json Python object
        for line in data_python:
            # writes a row and gets the fields from the json object
            # screen_name and followers/friends are found on the second level hence two get methods
            retweeted_status = False
            if line.get('retweeted_status'):
                retweeted_status = True
            writer.writerow([datetime.strptime(line.get('created_at'), '%a %b %d %H:%M:%S +0000 %Y'),
                             # line.get('text'),
                             line.get('full_text'),
                             line.get('user').get('screen_name'),
                             line.get('user').get('followers_count'),
                             line.get('user').get('friends_count'),
                             line.get('retweet_count'),
                             line.get('favorite_count'),
                             str(retweeted_status)])

    csvOutput.close()



date = "2018-07-03"
num_input_file = 7

rawDataPath = '../data/rawData/'
rawDataFileName = 'worldcup'
rawDataFileList = [rawDataPath + rawDataFileName + date + '-' + str(i) + '.txt' for i in range(1, num_input_file + 1)]

rawJSONInputPath = 'C:/Users/JaZz-/Documents/Dissertation/data/'
rawJSONInputFileName = 'worldcup'
rawJSONInputFileList = [rawJSONInputPath + rawJSONInputFileName + date + '-' + str(i) + '.txt' for i in range(1, num_input_file + 1)]

rawJSONOutputPath = 'data/rawDataFromJSON/'
rawJSONOutputFileName = 'worldcup'
rawJSONOutputFileName = rawJSONOutputPath + rawJSONOutputFileName + date + '.csv'

# txtToJSON(rawDataFileList)
# JSONToCsvExtraction(rawJSONInputFileList, rawJSONOutputFileName)