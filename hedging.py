
_hedging_seed_words = \
    [
        'alleged', 'allegedly',
        'apparently',
        'appear', 'appears',
        'claim', 'claims',
        'could',
        'evidently',
        'largely',
        'likely',
        'mainly',
        'may', 'maybe', 'might',
        'mostly',
        'perhaps',
        'presumably',
        'probably',
        'purported', 'purportedly',
        'reported', 'reportedly',
        'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
        'says',
        'seem',
        'somewhat',
        # 'supposedly',
        'unconfirmed']

_hedging_words = _hedging_seed_words

import numpy as np

def check_hedging_words(id,headline,body):

    hh_c = 0
    hb_c = 0
    for w in _hedging_words:
        if w in headline.lower().split():
            hh_c+=1
        if w in body.lower().split():
            hb_c += 1
    return [hh_c,hb_c]


print(check_hedging_words(0,"this could be it","something about rumours"))
