# Author: Giorgos Myrianthous

from utils.dataset import DataSet
from utils.generate_test_splits import split
from os import path
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pylab as py
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from tqdm import tqdm
from scipy import sparse
import csv, random, numpy, score, os, re, nltk, scipy, gensim
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from langdetect import detect
from sklearn.ensemble import RandomForestClassifier

dataset = DataSet()
lemmatizer = nltk.WordNetLemmatizer()

# Get the bodies of training data points
def get_bodies(data):
    bodies = []
    for i in range(len(data)):
        bodies.append(dataset.articles[data[i]['Body ID']])    
    return bodies

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
def extract_word_overlap(headlines, bodies):
    word_overlap = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        preprocess_headline = preprocess(headline)
        preprocess_body = preprocess(body)
        features = len(set(preprocess_headline).intersection(preprocess_body)) / float(len(set(preprocess_headline).union(preprocess_body)))
        word_overlap.append(features)

        # Convert the list to a sparse matrix (in order to concatenate the cos sim with other features)
        word_overlap_sparse = scipy.sparse.coo_matrix(numpy.array(word_overlap)) 

    return word_overlap_sparse

# Function for extracting tf-idf vectors (for both the bodies and the headlines).
def extract_tfidf(training_headlines, training_bodies, dev_headlines, dev_bodies, test_headlines, test_bodies):
    # Body vectorisation
    body_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')#, max_features=1024)
    bodies_tfidf = body_vectorizer.fit_transform(training_bodies)

    # Headline vectorisation
    headline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')#, max_features=1024)
    headlines_tfidf = headline_vectorizer.fit_transform(training_headlines)

    # Tranform dev/test bodies and headlines using the trained vectorizer (trained on training data)
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
def extract_cosine_similarity(headlines, bodies):
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
    cos_sim_array = scipy.sparse.coo_matrix(numpy.array(cos_sim_features)) 

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


# Function for combining features of various types (lists, coo_matrix, np.array etc.)
def combine_features(tfidf_vectors, cosine_similarity, word_overlap):
    combined_features =  sparse.bmat([[tfidf_vectors, word_overlap.T, cosine_similarity.T]])
    return combined_features

# Function for extracting features
# Feautres: 1) Word Overlap, 2) TF-IDF vectors, 3) Cosine similarity, 4) Word embeddings
def extract_features(train, dev, test):
    # Get bodies and headlines for dev and training data
    training_bodies = get_bodies(training_data)
    training_headlines = get_headlines(training_data)
    dev_bodies = get_bodies(dev_data)
    dev_headlines = get_headlines(dev_data)
    test_bodies = get_bodies(test_data)
    test_headlines = get_headlines(test_data)

    # Extract tfidf vectors
    print("\t-Extracting tfidf vectors..")
    training_tfidf, dev_tfidf, test_tfidf = extract_tfidf(training_headlines, training_bodies, dev_headlines, dev_bodies, test_headlines, test_bodies)


    # Extract word overlap 
    print("\t-Extracting word overlap..")
    training_overlap = extract_word_overlap(training_headlines, training_bodies)
    dev_overlap = extract_word_overlap(dev_headlines, dev_bodies)
    test_overlap = extract_word_overlap(test_headlines, test_bodies)

    # Extract cosine similarity between bodies and headlines. 
    print("\t-Extracting cosine similarity..")
    training_cos = extract_cosine_similarity(training_headlines, training_bodies)
    dev_cos = extract_cosine_similarity(dev_headlines, dev_bodies)
    test_cos = extract_cosine_similarity(test_headlines, test_bodies)

    # Combine the features
    training_features = combine_features(training_tfidf, training_cos, training_overlap)
    dev_features = combine_features(dev_tfidf, dev_cos, dev_overlap)
    test_features = combine_features(test_tfidf, test_cos, test_overlap)

    return training_features, dev_features, test_features

if __name__ == '__main__':
    ##############################################################################

    # Load the data
    print("\n[1] Loading data..")
    data_splits = split(dataset)

    # in the format: Stance, Headline, BodyID
    training_data = data_splits['training']
    dev_data = data_splits['dev']
    test_data = data_splits['test'] # currently 0 test points

    # Change the number of training examples used.
    N = int(len(training_data) * 1.0)
    training_data = training_data[:N]

    print("\t-Training size:\t", len(training_data))
    print("\t-Dev size:\t", len(dev_data))
    print("\t-Test data:\t", len(test_data))

    ##############################################################################

    # Feature extraction
    print("[2] Extracting features.. ")
    training_features, dev_features, test_features = extract_features(training_data, dev_data, test_data)

    ##############################################################################

    # Fitting model
    print("[3] Fitting model..")
    print("\t-Logistic Regression")
    lr = LogisticRegression(C = 1.0, class_weight='balanced', solver="lbfgs", max_iter=150) 
    #lr = RandomForestClassifier(n_estimators=10, random_state=12345)
    #lr = MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)

    targets_tr = [a['Stance'] for a in training_data]
    targets_dev = [a['Stance'] for a in dev_data]
    targets_test = [a['Stance'] for a in test_data]

    y_pred = lr.fit(training_features, targets_tr).predict(test_features)

    ##############################################################################

    # Evaluation
    print("[4] Evaluating model..")
    score.report_score(targets_test, y_pred)