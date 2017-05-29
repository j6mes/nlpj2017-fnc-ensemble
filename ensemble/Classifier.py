import pickle
import os

from tqdm import tqdm


class Classifier:
    def __init__(self,dataset):
        self.dataset = dataset
        self.fdict = dict()

    def predict(self,data):
        pass

    def train(self,data):
        pass

    def features(self,stance):
        return self.fdict[stance['Stance ID']],stance['Stance']

    def xys(self,data):
        Xs = []
        ys = []
        for datum in data:
            fts = self.features(datum)
            Xs.append(fts[0])
            ys.append(fts[1])

        return Xs,ys

    def load_feats(self,file,stances,ffns):
        if not os.path.isfile(file):
            fdict = self.gen_feats(stances,ffns)
            pickle.dump(fdict,open(file,"wb+"))

        return pickle.load(open(file,"rb"))

    def gen_feats(self, stances, ffns):

        fdict = dict()
        for stance in tqdm(stances):
            headline = stance['Headline']
            body = self.dataset.articles[stance['Body ID']]

            fs = []
            for ff in ffns:
                fs.extend(ff(headline,body))

            fdict[stance['Stance ID']] = fs

        return fdict