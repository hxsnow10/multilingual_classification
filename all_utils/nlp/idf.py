# IDF
from collections import defaultdict
import math

def get_features(x,vocab=None,split=' '):
    if type(x)==str:
        x=x.strip().split(split)
    if type(x)==list:
        x_=defaultdict(int)
        for word in x:
            if vocab is None or (type(vocab)==set and word in vocab):
                x_[word]+=1
        x=x_
    assert isinstance(x,dict)
    return x

def idf(X, vocab=None): 
    if vocab is not None:vocab=set(vocab)
    X=(get_features(x, vocab) for x in X)
    idf=defaultdict(float)
    N=0
    for x in X:
        N+=1
        for word in x.keys():
            idf[word]+=x[word]
    for x in idf:
        idf[x] = math.log((1+N)*1.0/(1+idf[x]))+1
    return idf

if __name__=="__main__":
    x=[' i love you .',' i hate you .', ' i love him .']
    print idf(x)
