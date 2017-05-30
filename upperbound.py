from utils.score import LABELS


def compute_ub(slaves,stances):
    actual = []
    for stance in stances:
        actual.append(LABELS.index(stance['Stance']))

    predicted = []
    for classifier in slaves:
        pred = classifier.predict(stances)
        pred = [LABELS.index(p) for p in pred]
        predicted.append(pred)

    oracle = 0
    predicted = list(zip(*predicted))
    for i,cls in enumerate(actual):
        if cls in predicted[i]:
            oracle += 1

    print(oracle)
    print(len(actual))

    print(oracle/len(actual))

