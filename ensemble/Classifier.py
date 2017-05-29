class Classifier:
    def predict(self,data):
        pass

    def train(self,data):
        pass

    def features(self,datum,name=""):
        pass

    def xys(self,data):
        Xs = []
        ys = []
        for datum in data:
            fts = self.features(datum)
            Xs.append(fts[0])
            ys.append(fts[1])

        return Xs,ys