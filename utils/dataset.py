from csv import DictReader


class DataSet():
    def __init__(self, ds="train" , path="fnc-1"):
        self.path = path

        print("Reading dataset")
        bodies = ds+"_bodies.csv"
        stances = ds+"_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        for sid,s in enumerate(self.stances):
            s['Stance ID'] = sid
            s['Body ID'] = int(s['Body ID'])
            if not 'Stance' in s.keys():
                s['Stance'] = "Unlabelled"

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))


    def read(self,filename):
        rows = []
        with open(self.path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
