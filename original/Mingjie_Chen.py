# coding=utf-8
from utils.dataset import DataSet
from utils.score import report_score
from collections import Counter
import re,sys
import numpy as np
import functools
import theano
import theano.tensor as T
import gzip
import pickle
import time
import keras
import gensim
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.layers import Input, Embedding, LSTM, Dense, Merge,\
    Bidirectional,Dropout,Convolution1D,Lambda,Concatenate,Flatten,MaxPooling1D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
#import lasagne
from keras import backend as K
from keras.models import Sequential

class Config:
    def __init__(self):
        self.debug = False
        self.max_example = 10000
        self.max_words = 15000
        self.embedding_file = "glove.6B/glove.6B.300d.txt"
        self.test_only = False
        self.model_file = "keras_model.pkl.gz"
        self.embedding_size = 300
        self.train_dropout_rate = 0.3
        self.vocab_size = 0
        self.hidden_size = 128
        self.rnn_output_size = 0
        self.pre_trained =None
        self.num_labels = 4
        self.num_train = 0
        self.num_dev = 0
        stopfile = open('Mingjie_Chen_stop_list.txt')
        self.stoplist = set([line.strip() for line in stopfile])


class process_data:


    """
    process data
    """

    def load_data(self, config):

        training_heads = []
        training_labels = []
        dev_heads = []
        dev_labels = []
        test_heads = []
        test_labels = []
        training_bodys = []
        dev_bodys = []
        test_bodys = []

        for n, dict in enumerate(training_data):

            if config.debug:
                if n <= config.max_example:
                    training_bodys.append(dataset.articles[dict["Body ID"]])
                    training_heads.append(dict.get("Headline"))
                    training_labels.append(dict.get("Stance"))
            else:
                training_bodys.append(dataset.articles[dict["Body ID"]])
                training_heads.append(dict.get("Headline"))
                training_labels.append(dict.get("Stance"))

        for dict in dev_data:
            dev_bodys.append(dataset.articles[dict["Body ID"]])

            dev_heads.append(dict.get("Headline"))
            dev_labels.append(dict.get("Stance"))
        for dict in test_data:
            test_bodys.append(dataset.articles[dict["Body ID"]])

            test_heads.append(dict.get("Headline"))
            test_labels.append(dict.get("Stance"))

        return (training_bodys, training_heads, training_labels), (dev_bodys, dev_heads
                            , dev_labels) , (test_bodys,test_heads, test_labels)

    def build_dict(self,train_examples,config):
        word_count = Counter()
        #word_2_ind = {}
        f = re.compile(r'([0-9a-zA-Z]+)')
        for d,q in list(zip(train_examples[0],train_examples[1])):

            for w in f.findall(d):
                if w.lower() not in config.stoplist:
                    word_count[w.lower()]+=1
            for w in f.findall(q):
                if w.lower() not in config.stoplist:
                    word_count[w.lower()]+=1


        # leave 0 to UNK
        print(len(word_count))
        word_2_ind = {w:index+1 for (index,w) in enumerate(word_count.keys())}
        return word_2_ind

    def gen_embeddings(self,word_dict, config):
        """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
        """

        num_words = max(word_dict.values()) + 1
        #num_words = len(word_dict)+1
        embeddings =np.zeros((num_words, config.embedding_size))
        print('Embeddings: %d x %d' % (num_words, config.embedding_size))
        word_vector =  gensim.models.KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300-small.bin", binary=True)
        for w in word_dict:
            if w in word_vector.vocab:
                embeddings[word_dict[w]] = word_vector[w]


        return embeddings

    def vectorize(self, examples, word_dict):
        """
        Vectorize `examples`.
        in_x1, in_x2: sequences for body and head  respecitvely.
        in_y: label

        """
        in_x1 = []
        in_x2 = []
        f = re.compile(r'([0-9a-zA-Z]+)')
        maxbody = 0
        maxtitle = 0
        for i, body in enumerate(examples[0]):

            d = body
            q = examples[1][i]
            d_words = f.findall(d)
            q_words = f.findall(q)

            seq1 = [word_dict[w.lower()] if w.lower() in word_dict else 0 for w in d_words]
            seq2 = [word_dict[w.lower()] if w.lower() in word_dict else 0 for w in q_words]
            if len(seq1) > maxbody:
                maxbody = len(seq1)
            if len(seq2) > maxtitle:
                maxtitle = len(seq2)
            if (len(seq1) > 0) and (len(seq2) > 0):
                in_x1.append(seq1)
                in_x2.append(seq2)


        return in_x1, in_x2




class MLP:
    def average_vector(self,x,y,embeddings):

        in_x = []
        for (temp_x,temp_y )in zip(x,y):
            body = [embeddings[tx] for tx in temp_x]
            title = [embeddings[ty] for ty in temp_y]
            body_vec = functools.reduce(lambda x,y: np.add(x,y),body )/len(body)
            title_vec = functools.reduce(lambda x,y:np.add(x,y),title)/len(title)

            in_x.append(np.concatenate((body_vec,title_vec)))
        return np.array(in_x)
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(256, activation='relu', input_dim=600))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(4, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    def train_dev(self,train_vec,train_l,dev_vec,dev_l):
        self.hist = self.model.fit(train_vec,train_l,batch_size=64,epochs=100
                       ,verbose=2,validation_data=(dev_vec,dev_l),shuffle=True)
        self.model.save_weights('features/mingjiechen.weights.h5')


    def train(self,train_vec,train_l):
        self.hist = self.model.fit(train_vec,train_l,batch_size=64,epochs=100,verbose=2,shuffle=True)



class Mingjie_Chen:
    def old__init__(self):
        config = Config()
        f = process_data()
        train_examples, dev_examples, test_examples = f.load_data(config)

        print("loading the data")
        config.num_train = len(train_examples[0])
        config.num_dev = len(dev_examples[0])
        print("training data is {}, dev data is {}, test data is  {}".format(config.num_train
                                                                             , config.num_dev
                                                                             , len(test_examples[0])))
        print("build word dict")
        word_dict = f.build_dict(train_examples, config)
        print("gen embedding")

        embeddings = f.gen_embeddings(word_dict, config)

        (config.vocab_size, config.embedding_size) = embeddings.shape

        train_x1, train_x2, self.train_y = f.vectorize(train_examples, word_dict)
        dev_x1, dev_x2, self.dev_y = f.vectorize(dev_examples, word_dict)
        test_x1, test_x2, self.test_y = f.vectorize(test_examples, word_dict)

        self.mlp = MLP()
        self.train_vec = self.mlp.average_vector(train_x1, train_x2, embeddings)
        self.dev_vec = self.mlp.average_vector(dev_x1, dev_x2, embeddings)
        self.test_vec = self.mlp.average_vector(test_x1, test_x2, embeddings)


    def train(self):
        print("train model")
        self.mlp.train(self.train_vec, self.train_y, self.dev_vec, self.dev_y)

    def predict(self):
        print("test model")
        self.mlp.model.load_weights('Mingjie_Chen_weights.h5')
        result = self.mlp.model.predict(self.test_vec)
        labels = ['unrelated', 'disagree', 'discuss', 'agree']
        result_lables = [labels[np.argmax(n)] for n in result]
        actual = [stance['Stance'] for stance in test_data]
        # actual = [labels[np.argmax(n)] for n in test_y]
        n = 0
        for a, p in zip(result_lables, actual):
            if a == p:
                n += 1
        print("test accuracy is {}".format(n / len(result_lables)))

        report_score(actual, result_lables)

if __name__=='__main__':
    cmj = Mingjie_Chen()
    cmj.predict()
'''
if __name__=='__main__':

    config = Config()
    f = process_data()
    train_examples, dev_examples, test_examples = f.load_data(config)


    print("loading the data")
    config.num_train = len(train_examples[0])
    config.num_dev = len(dev_examples[0])
    print("training data is {}, dev data is {}, test data is  {}".format(config.num_train
                                                                         , config.num_dev
                                                                         ,len(test_examples[0])))
    print("build word dict")
    word_dict = f.build_dict(train_examples, config)
    print("gen embedding")

    embeddings = f.gen_embeddings(word_dict, config)

    (config.vocab_size, config.embedding_size) = embeddings.shape

    train_x1, train_x2, train_y = f.vectorize(train_examples, word_dict)
    dev_x1, dev_x2, dev_y = f.vectorize(dev_examples, word_dict)
    test_x1,test_x2,test_y = f.vectorize(test_examples,word_dict)

    mlp = MLP()
    train_vec = mlp.average_vector(train_x1,train_x2,embeddings)
    dev_vec = mlp.average_vector(dev_x1,dev_x2,embeddings)
    test_vec = mlp.average_vector(test_x1,test_x2,embeddings)
    if_test = True

    if not if_test:
        print("train model")
        mlp.train(train_vec, train_y, dev_vec, dev_y)
        with open('history.pkl', 'wb') as output:
            pickle.dump(mlp.hist.history, output, pickle.HIGHEST_PROTOCOL)

    else:
        print("test model")
        mlp.model.load_weights('weights.h5')
        result = mlp.model.predict(test_vec)
        labels = ['unrelated','disagree','discuss','agree']
        result_lables = [labels[np.argmax(n)] for n in result]
        actual = [stance['Stance'] for stance in test_data]
        #actual = [labels[np.argmax(n)] for n in test_y]
        n = 0
        for a,p in zip(result_lables,actual):
            if a==p:
                n+=1
        print("test accuracy is {}".format(n/len(result_lables)))

        report_score(actual,result_lables)
'''












