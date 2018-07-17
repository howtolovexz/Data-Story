import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def getOriginalTweets(df):
    df_original = df.loc[df['retweeted_status'] == False]
    df_original = df_original.sort_values(by=['rt'], ascending=False)
    return df_original


def getRetweetedTweets(df):
    df_retweeted = df.loc[df['retweeted_status'] == True]
    df_retweeted = df_retweeted.sort_values(by=['rt'], ascending=False)
    return df_retweeted



def getTweetCountList(df):
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


def plotTweetsHistrogram(tweetCountList, title):
    pos = np.arange(len(tweetCountList))
    width = 1.0  # gives histogram aspect to the bar diagram

    ax = plt.axes()
    ax.grid(color='black', linestyle='--', linewidth=0.5)
    ax.set_xticks(pos + (width / 2))
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    ax.set_xticklabels([i for i in range(1, 25)])
    plt.yticks(np.arange(0, 120000, 10000))

    plt.xlabel('Hour')
    plt.ylabel('Number of Tweets')
    ax.set_title(title)

    plt.xlim([-0.5, 23.5])
    plt.bar(pos, [j[2] for j in tweetCountList], width, color='b', edgecolor='black')
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
    plt.yticks(np.arange(0, 120000, 10000))

    plt.xlabel('Hour')
    plt.ylabel('Number of Tweets')
    ax.set_title(title)

    plt.xlim([-0.5, 23.5])
    p1 = plt.bar(pos, [j[2] for j in tweetCountList1], width, color='b', edgecolor='black')
    p2 = plt.bar(pos, [j[2] for j in tweetCountList2], width, color='r', edgecolor='black', bottom=[j[2] for j in tweetCountList1])
    plt.show()

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def countTweetContainNumbers(df):
    count = 0
    for index, row in df.iterrows():
        if hasNumbers(row['text']):
            count = count + 1
    return count

date = "2018-06-30"
inputFileName = "data/rawDataFromJSON/worldcup" + date + ".csv"

colnames = ['created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status']
df = pd.read_csv(inputFileName, names=colnames)
df['created_at'] = pd.to_datetime(df['created_at'])  # change string to datetime
df_original = getOriginalTweets(df)
df_retweeted = getRetweetedTweets(df)

originalTweetCountList = getTweetCountList(df_original)
retweetedTweetCountList = getTweetCountList(df_retweeted)

# plotTweetsHistrogram(originalTweetCountList, 'Original Tweets histogram on 30 June 2018')
# plotTweetsHistrogram(retweetedTweetCountList, 'Retweeted Tweets histogram on 30 June 2018')
#
# plotStackedTweetsHistrogram(originalTweetCountList, retweetedTweetCountList, 'Tweets histogram on 03 July 2018')

# df_original.to_csv('data/OriginalTweets/worldcup' + date + 'original.csv', index=False, encoding='utf-8', header=False)

print(countTweetContainNumbers(df_original))
print(len(df_original))