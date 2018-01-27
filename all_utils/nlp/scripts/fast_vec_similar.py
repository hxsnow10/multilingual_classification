#!/usr/bin/env python
#encoding=utf-8
import numpy
import numpy as np
import faiss
import heapq
import sys

s=None

class W2vFast():
    '''for quick simikar search'''

    def __init__(self, w2v_path, index_type='fast'):
        self.word_vecs, _, self.d=self.load(w2v_path)
        self.build_index(self.word_vecs, self.d, index_type)

    def load(self,w2v_path):
        rval={}
        ii=open(w2v_path,'r')
        N,d=ii.readline().strip().split()
        N,d=int(N), int(d)
        for line in ii:
            s=line.strip().split()
            w=s[0]
            v=numpy.array(s[1:],dtype=numpy.float64)
            v=v/(np.sqrt(np.sum(v**2))+1)
            rval[w]=numpy.array(v,dtype=numpy.float32)
        ii.close()
        return rval,N,d

    def build_index(self, wv, d, index_type='fast', nlist=100):
        '''
        wv是一个字典,d是向量长度, 
        '''

        self.int2word={k:key for k,key in enumerate(wv.keys())}
        self.word2int={key:k for k,key in enumerate(wv.keys())}
        
        xb=numpy.array([v for v in wv.values() if len(v)==d])#不支持float16
        
        
        if index_type=='accurate':
            index = faiss.IndexFlatL2(d)
            index.add(xb)
        elif index_type=='fast':
            # nlist = 100# ?
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            print xb.shape
            index.train(xb)
            
            index.add(xb)
            #index.own_fields = False 
            self.quantizer=quantizer#必须加，否则这个东西会被回收，导致self.index悄悄地被修改了！ （搞了3个小时..， 问了作者)
        elif index_type=='compress':
            nlist = 64#?
            m = 8                      # number of bytes per vector
            quantizer = faiss.IndexFlatL2(d)  # this remains the same
            index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
            # 8 specifies that each sub-vector is encoded as 8 bits
            index.train(xb)
            index.add(xb)
            #index.own_fields = True 
            self.quantizer=quantizer
        self.index_type=index_type
        self.index_ = index
        self.xb=xb

    def index_search(self, v, k, nprobe=10):
        v=np.array([v],dtype=numpy.float32)
        if not nprobe:# ?
            nprobe=10
        self.index_.nprobe=nprobe
        print v,type(v), v.shape
        print k
        D,I=self.index_.search(v,k)
        return zip(list(I[0]),list(1-D[0]/2))

    def distance(self,w1,w2):
        '''
        cos similar of w1 & w2
        print w1,w2
        print w1 in self.word_vecs
        print w2 in self.word_vecs
        '''
        if not (w1 in self.word_vecs and w2 in self.word_vecs):return 0
        d=float(numpy.dot(numpy.array([self.word_vecs[w1]]),numpy.array([self.word_vecs[w2]]).T)[0][0])
        return d

    def most_similar(self, w, k, nprobe=10, context=''):
        if type(w)==type([1,2,3]):
            possible_words,scores=[],[]
            for word in w:
                words=self.most_similar(word,k)
                possible_words=possible_words+[w_ for w_,v in words]
            for ww in possible_words:
                score=0
                for word in w:
                    score+=self.distance(word,ww)
                score=score/len(w)
                scores.append(score)
            print possible_words
            rval=zip(possible_words, scores)
            rval=heapq.nlargest(k, rval, key=lambda x:x[1]) 
            return rval
        else:
            if w not in self.word_vecs:
                #v=self.extend(w,context)
                return []
            else:
                v=self.word_vecs[w]
            s=self.index_search(v,k,nprobe)
            words=[(self.int2word[i],float(j)) for i,j in s]
            return words

    def extend(self, w, context, update=False):
        # denpend on language model
        # return vector of w
        cv=sum(self.word_vecs[w] for w in seg(context) if w in self.word_vecs)
        return None

    def add_wv(self, word, vec):#Next Todo
        it=len(self.int2word)
        self.int2word[it]=word
        self.word2it[word]=it
        self.index.add(np.array([vec],dtype=numpy.float16))

if __name__=='__main__':
    #model=W2vFast('vec.txt','accurate')
    #model=W2vFast('vectors.bin')
    vec_path=sys.argv[1]
    model=W2vFast(vec_path)
    #for w in model.most_similar(['我'],10):
    #    print w
    #model.add_wv
    while True:
        try:
            word=raw_input('Input word:')
            k=raw_input('Input K:')
            word=[word.strip()]
            k=int(k.strip())
            for word,j in model.most_similar(word,k):
                print word,j
        except Exception,e:
            print e
