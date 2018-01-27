#!/usr/bin/env python
#encoding=utf-8
import sys
import codecs
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签  
#mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from nlp.word2vec import load_w2v
from nlp.synonym import load_synonym
from collections import OrderedDict 
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def main():
    
    embeddings_file = sys.argv[1]
    words_file = sys.argv[2]
    TOPN = int(sys.argv[3])
    languages = sys.argv[4].split('-')

    w2v = load_w2v(embeddings_file)
    synonym = load_synonym(words_file)
    synonym_=set()
    
    
    for w in synonym:
        if w not in w2v:continue
        for w_ in synonym[w]:
            if w_ in w2v and w!=w_:
                print w,w_
                synonym_.update([w,w_])
                break
                if len(synonym_)>TOPN:break
        if len(synonym_)>TOPN:break

    w2v = OrderedDict([(w,w2v[w]) for w in w2v if w in synonym_])
            
    #cp = TSNE(n_components=2, random_state=0)
    cp=PCA(2)
    np.set_printoptions(suppress=True)
    Y = cp.fit_transform(np.array(w2v.values()))
    print w2v.keys() 
    print Y.shape
    
    plt.scatter(Y[:, 0], Y[:, 1], marker='.', alpha=0.4)
    for label, x, y in zip(w2v.keys(), Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    #plt.show()
    plt.savefig("multilingual.jpg") 
 
'''
def load_embeddings(file_name):
 
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in 
f_in])
    wv = np.loadtxt(wv)
    return wv, vocabulary
'''

if __name__ == '__main__':
    main()
