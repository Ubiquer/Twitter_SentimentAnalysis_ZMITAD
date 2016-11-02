from tweepy import Stream
from tweepy import OAuthHandler
from py_stuff import *
import re
from tweepy.streaming import StreamListener
from nltk.corpus import stopwords





# class listener(StreamListener):
#
#     def on_data(self, data):
#         print(data)
#         return True
#     def on_error(self, status_code):
#         print(status_code)
#
#
# auth = OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_secret)
#
# twitterStream = Stream(auth, listener())
# twitterStream.filter(track=["Trump"])