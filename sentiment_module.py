from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
import random
from statistics import mode
import pickle
from nltk.tag import PerceptronTagger
from nltk.data import find

PICKLE = "averaged_perceptron_tagger.pickle"
AP_MODEL_LOC = 'file:'+str(find('taggers/averaged_perceptron_tagger/'+PICKLE))
tagger = PerceptronTagger(load=False)
tagger.load(AP_MODEL_LOC)
pos_tag = tagger.tag


import json
import urllib

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:    # c stands for classifier
            v = c.classify(features)   # for each classifier we are getting the vote
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes)) # how many occurances of most popular votes were in list of votes
        conf = choice_votes / len(votes)  # certainty
        print("funkcja classify, wartość: " + str(conf))
        return conf


documents_file = open("pickled_algorithms/documents.pickle", "rb")
documents = pickle.load(documents_file)
documents_file.close()

saved_word_features = open("pickled_algorithms/word_features.pickle", "rb")
load_word_features = pickle.load(saved_word_features)
saved_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in load_word_features:
        features[w] = (w in words)

    return features

featuresets_file = open("pickled_algorithms/featuresets.pickle", "rb")
load_featuresets = pickle.load(featuresets_file)
featuresets_file.close()


random.shuffle(load_featuresets)
print(len(load_featuresets))

testing_set = load_featuresets[10000:]
training_set = load_featuresets[:10000]

open_pickle = open("pickled_algorithms/naivebayes.pickle", "rb")
classifier = pickle.load(open_pickle)
open_pickle.close()

open_pickle = open("pickled_algorithms/BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_pickle)
open_pickle.close()

open_pickle = open("pickled_algorithms/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_pickle)
open_pickle.close()

open_pickle = open("pickled_algorithms/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_pickle)
open_pickle.close()

# open_pickle = open("pickled_algorithms/SGDC_classifier5k.pickle", "rb")
# SGDClassifier_classifier = pickle.load(open_pickle)
# open_pickle.close()

open_pickle = open("pickled_algorithms/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_pickle)
open_pickle.close()

open_pickle = open("pickled_algorithms/LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_pickle)
open_pickle.close()


voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  # SGDC_classifier5k,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  )


def sentiment(text):
    feats = find_features(text)
    print("funkcja sentiment txt: " + str(text))
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
