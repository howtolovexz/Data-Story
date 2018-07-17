import tweepy
import csv
import json
import time
import pandas as pd
from datetime import datetime

consumer_key = "0d5RbbKTfMEGHUi5XNtUhrmox"
consumer_secret = "FltNOhMkru9aIjYaAGmnRSkt7dwy5flo2ZOrUvwbZMjTQMu6eR"
access_token = "1004881161229930496-yurR9Y88MrSUAZil39o2OvF30buZCu"
access_token_secret = "gN92eFBUjbMfB3BfRPWzg8mZOSkaW9WENgjH57pkh5A7d"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

searchQuery = "#fifa OR #wc2018 OR #fifaworldcup OR #worldcup OR #russia2018"  # this is what we're searching for
maxTweets = 100000  # Some arbitrary large number
tweetsPerQry = 100  # this is the max the API permits
since_date = "2018-07-11"
until_date = "2018-07-12"
file_name = "../data/rawData/worldcup" + since_date + "-5.csv"
file_name2 = "../data/rawData/worldcup" + since_date + "-5.txt"
# Open/Create a file to append data
csvFile = open(file_name, 'a', encoding="utf-8")
jsonFile = open(file_name2, 'a', encoding="utf-8")
# Use csv Writer
csvWriter = csv.writer(csvFile)

# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = None

# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
# max_id = -1
max_id = 1017134916448989184

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))
while tweetCount < maxTweets:
    try:
        if (max_id <= 0):
            if (not sinceId):
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang="en", since=since_date, until=until_date, tweet_mode='extended')
            else:
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang="en", since_id=sinceId, since=since_date,
                                        until=until_date, tweet_mode='extended')
        else:
            if (not sinceId):
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang="en", max_id=str(max_id - 1), since=since_date,
                                        until=until_date, tweet_mode='extended')
            else:
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang="en", max_id=str(max_id - 1), since_id=sinceId,
                                        since=since_date, until=until_date, tweet_mode='extended')
        if not new_tweets:
            print("No more tweets found")
            new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang="en", max_id=str(max_id - 1), since=since_date,
                                    until=until_date, tweet_mode='extended')
            if not new_tweets:
                break
        for tweet in new_tweets:
            csvWriter.writerow([tweet.created_at, tweet.full_text.lower(), tweet.retweet_count, tweet.source])
            json.dump(tweet._json, jsonFile, sort_keys=True, indent=4, ensure_ascii=False)
        tweetCount += len(new_tweets)
        data = api.rate_limit_status()
        # print(data['resources']['search']['/search/tweets'])
        print("Downloaded {0} tweets".format(tweetCount))
        max_id = new_tweets.max_id + 1
        print("max_id: " + str(max_id))
    except tweepy.TweepError as e:
        # Just exit if any error
        print("some error : " + str(e))
        break

csvFile.close()
jsonFile.close()
print("DONE!")