from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

# consumer key, consumer secret, access token, access secret.
consumer_key = "VSUjSwlyHxntKURAtAV9JhUk3"
consumer_secret = "F7yDGOEAYyQqIZcqJENEzJcX7tz00lnvE6JPs2KY89H8Sd5jok"
access_token = "1004881161229930496-pgBkpPEtcbEUMNxPtrogcmuVFNxwgG"
access_token_secret = "6Bfi3eZ55doDKewzMa0sVjIB3ktYg5F2NiMQgDiiFgpeY"

class listener(StreamListener):

    def on_data(self, data):
        print(data)
        return (True)

    def on_error(self, status):
        print
        status


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["#worldcup"])
