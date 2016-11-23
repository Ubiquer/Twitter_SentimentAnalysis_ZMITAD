import pickle
import random
import nltk
import numpy
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
from statistics import mode
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

        return conf

short_pos = open('short_reviews/positive.txt', 'r').read()
short_neg = open('short_reviews/negative.txt', 'r').read()

documents = []
all_words = []

part_of_speech_allowed = ["J"]

for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in part_of_speech_allowed:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in part_of_speech_allowed:
            all_words.append(w[0].lower())

save_documents = open("pickled_algorithms/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

# save_word_features = open("pickled_algorithms/word_features5k.pickle", "wb")
# pickle.dump(word_features, save_word_features)
# save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)

testing_set = featuresets[10000:]
training_set = featuresets[:10000]

classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier.show_most_informative_features(15)
print("Bayes classifier accuracy:", (nltk.classify.accuracy(classifier, testing_set))*100)

# save_classifier = open("pickled_algorithms/originalnaivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# save_classifier = open("pickled_algorithms/MNB_classifier.pickle","wb")
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier accuracy:", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
# save_classifier = open("pickled_algorithms/BernoulliNB_classifier.pickle", "wb")
# pickle.dump(BernoulliNB_classifier, save_classifier)
# save_classifier.close()
print("BernoulliNB_classifier accuracy:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
# save_classifier = open("pickled_algorithms/SGDC_classifier5k.pickle","wb")
# pickle.dump(SGDClassifier_classifier, save_classifier)
# save_classifier.close()
print("SGDC_classifier accuracy:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
# save_classifier = open("pickled_algorithms/LinearSVC_classifier.pickle", "wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()
print("LinearSVC_classifier accuracy:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier accuracy:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)


# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()


voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  SGDClassifier_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier,
                                  )

print("voted_classifier accuracy: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
# print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence :", voted_classifier.confidence(testing_set[0][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence :", voted_classifier.confidence(testing_set[1][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[2][0]), "Confidence :", voted_classifier.confidence(testing_set[2][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence :", voted_classifier.confidence(testing_set[3][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence :", voted_classifier.confidence(testing_set[4][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[5][0]), "Confidence :", voted_classifier.confidence(testing_set[5][0])*100)