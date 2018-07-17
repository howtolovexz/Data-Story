import tweepy
import csv
import time
import pandas as pd
from datetime import datetime


# def limit_handled(cursor):
#     while True:
#         try:
#             yield cursor.next()
#         except tweepy.RateLimitError:
#             print("Exceed Limit Rate")
#             print("Start sleeping: " + str(datetime.now()))
#             time.sleep(15 * 60)
#             print("End sleeping: " + str(datetime.now()))

def limit_handled():
    data = api.rate_limit_status()
    print(data["resources"]["search"]["/search/tweets"])
    if tweepy.RateLimitError:
        print("Exceed Limit Rate")
        print("Start sleeping: " + str(datetime.now()))
        time.sleep(15 * 60)
        print("End sleeping: " + str(datetime.now()))


consumer_key = "0d5RbbKTfMEGHUi5XNtUhrmox"
consumer_secret = "FltNOhMkru9aIjYaAGmnRSkt7dwy5flo2ZOrUvwbZMjTQMu6eR"
access_token = "1004881161229930496-yurR9Y88MrSUAZil39o2OvF30buZCu"
access_token_secret = "gN92eFBUjbMfB3BfRPWzg8mZOSkaW9WENgjH57pkh5A7d"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

since_date = "2018-06-08"
# file_name = "worldcup" + since_date + ".csv"
# file_name = "test.csv"
# Open/Create a file to append data
# csvFile = open(file_name, 'a', encoding="utf-8")
# Use csv Writer
# csvWriter = csv.writer(csvFile)
searchQuery = "the first win for the african representatives..."

print("Start: " + str(datetime.now()))
last_id = -1
for i in range(1, 2):
    print("Round: " + str(i))
    print("Time: " + str(datetime.now()))
    search_results = tweepy.Cursor(api.search, q=searchQuery, count=15, lang="en", since="2018-06-16",
                                   until="2018-06-17", since_id=str(last_id)).items(10)

    for tweet in search_results:
        print(tweet)
        # if (not tweet.retweeted) and ('RT @' not in tweet.text):
        # csvWriter.writerow([tweet.created_at, tweet.text.lower()])

    print(search_results.page_iterator.max_id)
    last_id = search_results.current_page.max_id
    print("Round: " + str(i) + " Done!")

# csvFile.close()
print("Finish: " + str(datetime.now()))
