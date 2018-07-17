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

maxTweets = 100  # Some arbitrary large number
tweetsPerQry = 100  # this is the max the API permits
since_date = "2018-06-27"
until_date = "2018-06-28"
searchQuery = "#fifa OR #wc2018 OR #fifaworldcup OR #worldcup OR #russia2018"  # this is what we're searching for

file_name = "topTweets" + since_date + ".csv"
file_name2 = "topTweets" + since_date + ".txt"
# Open/Create a file to append data
csvFile = open(file_name, 'a', encoding="utf-8")
jsonFile = open(file_name2, 'a', encoding="utf-8")
# Use csv Writer
csvWriter = csv.writer(csvFile)

print("Start")
search_results = tweepy.Cursor(api.search, q=searchQuery, count=100, lang="en", since=since_date,
                               until=until_date, result_type='popular').items(100)

for tweet in search_results:
    print("TEXT: " + tweet.text)
    print("Retweet count: " + str(tweet.retweet_count))
    print(tweet.user.name)

    csvWriter.writerow([tweet.created_at, tweet.text.lower(), tweet.retweet_count, tweet.source, tweet.user.name])
    json.dump(tweet._json, jsonFile, sort_keys=True, indent=4, ensure_ascii=False)

csvFile.close()
jsonFile.close()
print("DONE!")

