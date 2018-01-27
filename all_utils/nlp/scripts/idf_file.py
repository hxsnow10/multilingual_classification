#!/usr/bin/env python
# encoding=utf-8
from collections import defaultdict, OrderedDict
from nlp.idf import idf 
from nlp.word2vec import save_w2v
import argparse
import numpy as np
def data(paths):
    for path in paths:
        ii=open(path, 'r')
        for line in ii:
            yield line
        
def main(file_paths, output_path, min_count=10):
    vocab_=defaultdict(int)
    for line in data(file_paths):
        for word in line.strip().split():
            vocab_[word]+=1
    vocab=defaultdict(int)

    for word in vocab_:
        if vocab_[word]>=min_count:
            vocab[word] = vocab_[word]
    vocab=vocab.keys()
    d_idf = OrderedDict((w,np.array([v])) for w,v in\
            sorted(idf(data(file_paths), vocab).iteritems(),key=lambda x:-x[1]))
    save_w2v(d_idf, output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_paths', nargs='+')
    parser.add_argument('--output_path')
    parser.add_argument('--min_count', default=10, type=int)
    args = parser.parse_args()
    main(args.file_paths, args.output_path, args.min_count)

