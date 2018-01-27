#encoding=utf-8
from collections import defaultdict

def count_words(paths):
    if type(paths)==str:
        paths=[paths]
    d=defaultdict(int)
    for path in paths:
        ii=open(path)
        for line in ii:
            line=line.strip().replace('\t',' ')
            for word in line.split():
                d[word]+=1
    return d

def get_words(paths, min_count=None):
    d=count_words(paths)
    new_d={}
    if min_count:
        for w in d:
            if d[w]>=min_count:
                new_d[w]=d[w]
    else:
        new_d=d
    return new_d
