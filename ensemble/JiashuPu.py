from ensemble.Classifier import Classifier
from original.jp import FncClassifier, Feature


class JiashuPu(Classifier):
    def __init__(self,data,train):
        super().__init__(data,train)
        self.jp = FncClassifier()
        word2vector = 'GoogleNews-vectors-negative300-small.bin'
        feature_generator = Feature(word2vector)
        self.jp.load_feature_generator(feature_generator)
        self.jp.set_mlp((1010, 6))


    def train(self,data):
        Xs,ys = self.xys(data)
        self.jp.mlp_clf.fit(Xs,ys)

    def predict(self,data):
        Xs,ys = self.xys(data)
        prd = self.jp.mlp_clf.predict(Xs)
        return prd

    def load_w2v(self):
        word2vector = 'GoogleNews-vectors-negative300-small.bin'
        feature_generator = Feature(word2vector)
        self.jp.load_feature_generator(feature_generator)

    def delete_big_files(self):
        super().delete_big_files()
        del self.jp.feature_generator

    def preload_features(self,data,fext=""):
        if not hasattr(self,'fdict'):
            self.fdict = dict()
        self.fdict.update(self.load_feats("features/jp."+fext+"pickle",data,list([self.jp.get_feature_for_stance])))