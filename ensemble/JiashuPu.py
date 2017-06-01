from ensemble.Classifier import Classifier
from ensemble.jp import FncClassifier, Feature


class JiashuPu(Classifier):
    def __init__(self,data):
        super().__init__(data)
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

    def preload_features(self,data):
        self.fdict = self.load_feats("features/jp.pickle",data,list([self.jp.get_feature_for_stance]))