# Author: Giorgos Myrianthous
from ensemble.Classifier import Classifier
from utils.dataset import DataSet
from os import path
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from tqdm import tqdm
from scipy import sparse
import csv, random, numpy, os, re, nltk, scipy, gensim
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from langdetect import detect
from sklearn.ensemble import RandomForestClassifier
import numpy as np
lemmatizer = nltk.WordNetLemmatizer()


# Get the headlines of training data points
def get_headlines(data):
    headlines = []
    for i in range(len(data)):
        headlines.append(data[i]['Headline'])
    return headlines

# Tokenisation, Normalisation, Capitalisation, Non-alphanumeric removal, Stemming-Lemmatization
def preprocess(string):
    # to lowercase, non-alphanumeric removal
    step1 = " ".join(re.findall(r'\w+', string, flags=re.UNICODE)).lower()
    step2 = [lemmatizer.lemmatize(t).lower() for t in nltk.word_tokenize(step1)]

    return step2


# Function for extracting word overlap
def extract_word_overlap(id,headline, body):
    preprocess_headline = preprocess(headline)
    preprocess_body = preprocess(body)
    features = len(set(preprocess_headline).intersection(preprocess_body)) / float(len(set(preprocess_headline).union(preprocess_body)))
    return [features]

# Function for extracting tf-idf vectors (for both the bodies and the headlines).
def extract_tfidf(training_headlines, training_bodies, dev_headlines, dev_bodies, test_headlines, test_bodies):
    # Body vectorisation
    body_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')#, max_features=1024)
    bodies_tfidf = body_vectorizer.fit_transform(training_bodies)

    # Headline vectorisation
    headline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')#, max_features=1024)
    headlines_tfidf = headline_vectorizer.fit_transform(training_headlines)

    # Tranform dev/test bodies and headlines using the trained vectori47zer (trained on training data)
    bodies_tfidf_dev = body_vectorizer.transform(dev_bodies)
    headlines_tfidf_dev = headline_vectorizer.transform(dev_headlines)

    bodies_tfidf_test = body_vectorizer.transform(test_bodies)
    headlines_tfidf_test = headline_vectorizer.transform(test_headlines)

    # Combine body_tfdif with headline_tfidf for every data point. 
    training_tfidf = scipy.sparse.hstack([bodies_tfidf, headlines_tfidf])
    dev_tfidf = scipy.sparse.hstack([bodies_tfidf_dev, headlines_tfidf_dev])
    test_tfidf = scipy.sparse.hstack([bodies_tfidf_test, headlines_tfidf_test])

    return training_tfidf, dev_tfidf, test_tfidf

# Function for extracting the cosine similarity between bodies and headlines. 
def extract_cosine_similarity(id, headline, body):
    bodies = [body]
    headlines = [headline]

    vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True, stop_words='english')#, max_features=1024)

    cos_sim_features = []
    for i in range(0, len(bodies)):
        body_vs_headline = []
        body_vs_headline.append(bodies[i])
        body_vs_headline.append(headlines[i])
        tfidf = vectorizer.fit_transform(body_vs_headline)

        cosine_similarity = (tfidf * tfidf.T).A
        cos_sim_features.append(cosine_similarity[0][1])

    # Convert the list to a sparse matrix (in order to concatenate the cos sim with other features)
    cos_sim_array = numpy.array(cos_sim_features)

    return cos_sim_array


# Function for counting words
def extract_word_counts(headlines, bodies):
    word_counts = []

    for i in range(0, len(headlines)):
        features = []
        features.append(len(headlines[i].split(" ")))
        features.append(len(bodies[i].split(" ")))
        word_counts.append(features)
    word_counts_array = scipy.sparse.coo_matrix(numpy.array(word_counts))

    return word_counts_array 





class GiorgosMyrianthous(Classifier):
    def __init__(self,data,training_data):
        super().__init__(data,training_data)
        self.lr = LogisticRegression(C=1.0, class_weight='balanced', solver="lbfgs", max_iter=150, random_state=124912)
        self.training_data = training_data

        training_bodies = []
        training_headlines = []
        for stance in training_data:
            training_headlines.append(stance['Headline'])
            training_bodies.append(data.articles[stance['Body ID']])

        # Body vectorisation
        self.body_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')
        self.body_vectorizer.fit(training_bodies)

        # Headline vectorisation
        self.headline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')
        self.headline_vectorizer.fit(training_headlines)

    def train(self,data):
        Xs,ys = self.xys(data)
        Xs = scipy.sparse.vstack(Xs)
        self.lr.fit(Xs,ys)


    def predict(self,data):
        Xs,ys = self.xys(data)
        Xs = scipy.sparse.vstack(Xs)
        pred = self.lr.predict(Xs)
        return pred


    def f_tfidf(self,id,headline,body):
        bodies_tfidf = self.body_vectorizer.transform([body])
        headlines_tfidf = self.headline_vectorizer.transform([headline])
        return scipy.sparse.hstack((bodies_tfidf, headlines_tfidf))


    def preload_features(self,stances,fext=""):
        if not hasattr(self,'fdict'):
            self.fdict = dict()
        ff = [self.f_tfidf,extract_cosine_similarity, extract_word_overlap]
        self.fdict.update(self.load_feats("features/gm."+fext+"pickle", stances,ff))

    def gen_feats(self, stances, ffns):
        print("Sparse feature generation")
        fdict = dict()
        for stance in tqdm(stances):
            headline = stance['Headline']
            body = self.dataset.articles[stance['Body ID']]

            fs = scipy.sparse.coo_matrix((0,1))
            for ff in ffns:
                fs = scipy.sparse.hstack((fs,ff(stance['Stance ID'],headline,body)))

            fdict[stance['Stance ID']] = fs

        return fdict