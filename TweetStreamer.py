from tweepy import Stream
from tweepy import OAuthHandler
from py_stuff import *
import re
from tweepy.streaming import StreamListener
import json
import sentiment_module as s
from nltk.corpus import stopwords





class listener(StreamListener):

    def on_data(self, data):

        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence*100 >= 60:
            output = open("twitter.txt", "a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
        return True

    def on_error(self, status_code):
        print(status_code)


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["Trump"])