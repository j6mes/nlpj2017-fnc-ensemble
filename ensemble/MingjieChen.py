import re

from ensemble.Classifier import Classifier
from original.Mingjie_Chen import MLP, process_data, Config
import numpy as np

from utils.score import LABELS


class MingjieChen(Classifier):
    def __init__(self,data,train):
        super().__init__(data,train)
        self.mlp = MLP()
        self.f = process_data()
        self.config = Config()

        headlines = []
        bodies = []

        for stance in train:
            headlines.append(stance['Headline'])
            bodies.append(data.articles[stance['Body ID']])

        self.word_dict = self.f.build_dict(list(zip(headlines,bodies)), self.config)
        self.embeddings = self.f.gen_embeddings(self.word_dict, self.config)

    def delete_big_files(self):

        self.mlp.model.save_weights('features/weights.h5')
        del self.f
        del self.mlp
        del self.config


    def load_w2v(self):
        self.f = process_data()
        self.config = Config()
        self.embeddings = self.f.gen_embeddings(self.word_dict, self.config)
        self.mlp = MLP()
        self.mlp.model.load_weights('features/weights.h5')

    def train(self,data):
        Xs,ys = self.xys(data)

        in_y = []
        for a in ys:
            if a == LABELS[0]:
                in_y.append([1, 0, 0, 0])
            elif a == LABELS[1]:
                in_y.append([0, 1, 0, 0])
            elif a == LABELS[2]:
                in_y.append([0, 0, 1, 0])
            elif a == LABELS[3]:
                in_y.append([0, 0, 0, 1])

        self.mlp.train(Xs,np.array(in_y))

    def predict(self,data):
        Xs,ys = self.xys(data)

        result = self.mlp.model.predict(Xs)
        labels = list(range(4))
        return [LABELS[labels[np.argmax(n)]] for n in result]

    def preload_features(self,data,fext=""):
        if not hasattr(self,'fdict'):
            self.fdict = dict()

        self.fdict.update(self.load_feats("features/mc."+fext+"pickle",data,[self.avg_embedding_lookup]))

    def avg_embedding_lookup(self,id,headline,body):
        return self.mlp.average_vector([self.wvec(headline)],[self.wvec(body)],self.embeddings)[0]

    def wvec(self,text):
        f = re.compile(r'([0-9a-zA-Z]+)')
        text = f.findall(text)
        return [self.word_dict[w.lower()] if w.lower() in self.word_dict else 0 for w in text]

