import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def getOriginalTweets(df):  # get dataframe with original tweet only
    df_original = df.loc[df['retweeted_status'] == False]
    # df_original = df_original.sort_values(by=['rt'], ascending=False)
    return df_original


def getRetweetedTweets(df):  # get dataframe with retweeted tweet only
    df_retweeted = df.loc[df['retweeted_status'] == True]
    # df_retweeted = df_retweeted.sort_values(by=['rt'], ascending=False)
    return df_retweeted


def getTweetCountList(df):  # get tweet count by hour
    tweetCountList = []
    date = df['created_at'].iloc[0]
    dateStr = date.strftime('%Y-%m-%d')
    for i in range(0, 24):
        endTime = str(i) + ':59:59'
        startTime = str(i) + ':00:00'
        if i < 10:
            endTime = '0' + endTime
            startTime = '0' + startTime
        # print(startTime + "-" + endTime)

        tweetCountList.append([startTime, endTime, len(df.loc[
                                                           (df['created_at'] <= datetime.strptime(
                                                               dateStr + ' ' + endTime, '%Y-%m-%d %H:%M:%S')) &
                                                           (df['created_at'] > datetime.strptime(
                                                               dateStr + ' ' + startTime, '%Y-%m-%d %H:%M:%S'))])])
    return tweetCountList


def getTweetCountList2(df):
    tweetCountList = []
    date = df['created_at'].iloc[0]
    dateStr = date.strftime('%Y-%m-%d')
    for i in range(0, 24):
        endTime = str(i) + ':59:59'
        startTime = str(i) + ':00:00'
        if i < 10:
            endTime = '0' + endTime
            startTime = '0' + startTime
        # print(startTime + "-" + endTime)

        tweetCountList.append(len(df.loc[
                                       (df['created_at'] <= datetime.strptime(
                                           dateStr + ' ' + endTime, '%Y-%m-%d %H:%M:%S')) &
                                       (df['created_at'] > datetime.strptime(
                                           dateStr + ' ' + startTime, '%Y-%m-%d %H:%M:%S'))]))
    return tweetCountList

def plotTweetsHistrogram(tweetCountList, title):
    pos = np.arange(len(tweetCountList))
    width = 1.0  # gives histogram aspect to the bar diagram

    ax = plt.axes()
    ax.grid(color='black', linestyle='--', linewidth=0.5)
    ax.set_xticks(pos + (width / 2))
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    ax.set_xticklabels([i for i in range(1, 25)])

    plt.xlabel('Hour')
    plt.ylabel('Number of Tweets')
    ax.set_title(title)

    plt.xlim([-0.5, 23.5])
    plt.bar(pos, [j[2] for j in tweetCountList], width, color='b', edgecolor='black')
    plt.yticks(np.arange(0, 310000, 20000))
    plt.show()


def plotStackedTweetsHistrogram(tweetCountList1, tweetCountList2, title):
    pos = np.arange(len(tweetCountList1))
    width = 1.0  # gives histogram aspect to the bar diagram

    ax = plt.axes()
    ax.grid(color='black', linestyle='--', linewidth=0.5)
    ax.set_xticks(pos + (width / 2))
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    ax.set_xticklabels([i for i in range(1, 25)])

    plt.xlabel('Hour')
    plt.ylabel('Number of Tweets')
    ax.set_title(title)

    plt.xlim([-0.5, 23.5])
    p1 = plt.bar(pos, [j[2] for j in tweetCountList1], width, color='#1a75ff', edgecolor='black')
    p2 = plt.bar(pos, [j[2] for j in tweetCountList2], width, color='#ff9933', edgecolor='black',
                 bottom=[j[2] for j in tweetCountList1])
    plt.yticks(np.arange(0, 80000, 5000))
    plt.show()
    plt.subplots_adjust(left=0.17, right=0.92, top=0.92, bottom=0.12)
    plt.savefig(title + '.png')
    plt.close()


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def countTweetContainNumbers(df):
    count = 0
    for index, row in df.iterrows():
        if hasNumbers(row['text']):
            count = count + 1
    return count


def removeNonNumberTweet(df):
    df_temp = df.loc[df.iloc[:, 1].str.contains(r'\d')]
    return df_temp


def removeNumberTweet(df):
    df_temp = df.loc[~df.iloc[:, 1].str.contains(r'\d')]
    return df_temp


def excludeLink(df):
    df = df.replace(regex=r"http\S+", value='')
    return df

def writeOutputFileFromList(outputFileName, dataList):
    outputFile = open(outputFileName, 'w', encoding="utf-8")
    csvWriter = csv.writer(outputFile)
    for dateCount in dataList:
        csvWriter.writerow([dateCount[0]] + dateCount[1])
    outputFile.close()

dateWithMatchList = ['2018-06-30', '2018-07-01', '2018-07-02', '2018-07-03', '2018-07-06', '2018-07-07', '2018-07-10',
                     '2018-07-11', '2018-07-14', '2018-07-15']  # match
dateWithoutMatchList = ['2018-07-04', '2018-07-05', '2018-07-08', '2018-07-09',
                        '2018-07-12', '2018-07-13']  # no match
dateList = dateWithMatchList + dateWithoutMatchList
inputFileList = ['../../data/rawDataFromJSON/worldcup' + date + '.csv' for date in dateList]

originalWithNumberList = []
originalWithoutNumberList = []
retweetWithNumberList = []
retweetWithoutNumberList = []
for inputFileName in inputFileList:
    colnames = ['created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status']
    df = pd.read_csv(inputFileName, names=colnames)
    df['created_at'] = pd.to_datetime(df['created_at'])  # change string to datetime
    df_original = getOriginalTweets(df)
    df_retweeted = getRetweetedTweets(df)

    # originalTweetCountList = getTweetCountList(df_original)
    # retweetedTweetCountList = getTweetCountList(df_retweeted)

    date = inputFileName[35:45]
    date = str(datetime.strptime(date, '%Y-%m-%d').strftime('%d %b %Y'))
    # plotTweetsHistrogram(originalTweetCountList, 'Original Tweets histogram on ' + date)
    # plotTweetsHistrogram(retweetedTweetCountList, 'Retweeted Tweets histogram on ' + date)

    # plotStackedTweetsHistrogram(originalTweetCountList, retweetedTweetCountList, 'Tweets histogram on ' + date)

    df_nonLink = excludeLink(df_original)
    df_numbers = removeNonNumberTweet(df_nonLink)
    df_nonnumbers = removeNumberTweet(df_nonLink)
    df_stat = df_numbers.loc[df_numbers.iloc[:, 1].str.contains(r'\bstat|record\b')]
    df_nostat = df_numbers.loc[~df_numbers.iloc[:, 1].str.contains(r'\bstat|record\b')]

    originalWithNumberList.append([date, getTweetCountList2(df_numbers)])
    originalWithoutNumberList.append([date, getTweetCountList2(df_nonnumbers)])

    df_nonLink = excludeLink(df_retweeted)
    df_numbers = removeNonNumberTweet(df_nonLink)
    df_nonnumbers = removeNumberTweet(df_nonLink)
    df_stat = df_numbers.loc[df_numbers.iloc[:, 1].str.contains(r'\bstat|record\b')]
    df_nostat = df_numbers.loc[~df_numbers.iloc[:, 1].str.contains(r'\bstat|record\b')]

    retweetWithNumberList.append([date, getTweetCountList2(df_numbers)])
    retweetWithoutNumberList.append([date, getTweetCountList2(df_nonnumbers)])

    # numberCountList = getTweetCountList(df_numbers)
    # nonnumverCountList = getTweetCountList(df_nonnumbers)
    # plotStackedTweetsHistrogram(numberCountList, nonnumverCountList, 'Tweets histogram on ' + date)
    # df_nostat.to_csv('../../data/StatisticTweets/worldcup' + date + 'nostatistic.csv', index=False, encoding='utf-8',
    #                  header=False)

# df_original.to_csv('../../data/OriginalTweets/worldcup' + date + 'original.csv', index=False, encoding='utf-8', header=False)
# df_numbers.to_csv('../../data/NumbersTweets/worldcup' + date + 'numbers.csv', index=False, encoding='utf-8', header=False)
# df_nonnumbers.to_csv('../../data/NumbersTweets/worldcup' + date + 'nonnumbers.csv', index=False, encoding='utf-8', header=False)
# df_stat.to_csv('../../data/StatisticTweets/worldcup' + date + 'statistic.csv', index=False, encoding='utf-8', header=False)

outputFileName = '../../data/test/originalWithNumber.csv'
writeOutputFileFromList(outputFileName, originalWithNumberList)
outputFileName = '../../data/test/originalWithoutNumber.csv'
writeOutputFileFromList(outputFileName, originalWithoutNumberList)
outputFileName = '../../data/test/retweetWithNumber.csv'
writeOutputFileFromList(outputFileName, retweetWithNumberList)
outputFileName = '../../data/test/retweetWithoutNumber.csv'
writeOutputFileFromList(outputFileName, retweetWithoutNumberList)