#encoding=utf-8
'''
generate source and target tensor from data
'''
import numpy as np
import Queue
import random
from functools import wraps
import json

from utils.base import dict_reverse
from utils.wraps import tryfunc
from utils import byteify

import config
class Vocab():

    def __init__(self, words={}, vocab_path=None, unk='</s>'):
        if words:
            self.vocab=words
            self.reverse_vocab = dict_reverse(words)
        else:
            self.vocab, self.reverse_vocab=self.get_vocab(vocab_path)
        if unk:
            self.unk=unk
            if self.unk not in self.reverse_vocab:
                k=len(self.vocab)
                self.vocab[k]=self.unk
                self.reverse_vocab[self.unk]=k

    def get_vocab(self, vocab_path):
        ii=open(vocab_path, 'r')
        vocab,reverse_vocab={},{}
        for line in ii:
            word=line.strip()
            k=len(vocab)
            vocab[word]=k
            reverse_vocab[k]=word
        return vocab, reverse_vocab

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, x):
        return self.reverse_vocab[x]

    def __contains__(self, x):
        return x in self.reverse_vocab

class label_line_processing():

    def __init__(self, words, split=' '):
        self.vocab=Vocab(words=words, unk=None)

    def __call__(self, line):
        words = line.strip().split()
        rval=[0,]*len(self.vocab)
        for word in words:
            if word in self.vocab:
                rval[self.vocab[word]]=1
        return [rval]

class data_line_processing():
    def __init__(self, max_len=100, pad=0):
        self.max_len=max_len
        self.pad=0
        self.size=1

    def __call__(self, line):
        if type(line)==str:
            data=line.strip().split()
            data=[eval(x) for x in data]
        else:
            data=line
        data=data[:self.max_len]
        #print data
        if len(data)<self.max_len:
            data=data+[self.pad,]*(self.max_len-len(data))
        return [data]

class sequence_line_processing():
    
    def __init__(self, words, max_len=100, split=' ', return_length=True, skip=True):
        self.vocab= Vocab(words=words)
        self.max_len =max_len
        self.split=split
        self.return_length=return_length
        self.size=1 if not return_length else 2
        self.skip=skip

    def __call__(self, line):
        if type(line)==str:
            words = line.strip().split(self.split)
        elif type(line)==list:
            words = line
        indexs=[]
        num=0
        for word in words:
            if word in self.vocab:
                indexs.append(self.vocab[word])
                num+=1
            elif not self.skip:
                indexs.append(self.vocab[self.vocab.unk])
            if len(indexs)>=self.max_len:
                break
        l=len(indexs)
        #print num*1.0/len(words)
        if len(indexs)<self.max_len:
            indexs=indexs+(self.max_len-len(indexs))*[self.vocab[self.vocab.unk],]
        if self.max_len!=1:
            item=np.array(indexs, dtype=np.int32)
        else:
            item=indexs[0]
        if not self.return_length:
            return [item]
        return [item, l]

class sequence_line_topn_processing():

    def __init__(self, words, topn=100, weights=None):
        self.vocab = Vocab(words=words)
        self.topn = topn

    def __call__(self, line):
        if type(line)==str:
            words = line.strip().split(self.split)
        elif type(line)==list:
            words = line
        rval={}
        pass

class split_line_processing():

    def __init__(self, line_processors=[], split='\t'):
        self.line_processors = line_processors
        self.size = len(line_processors)
        self.split='\t'

    def __call__(self, line):
        parts = line.strip().split(self.split)
        if len(parts)>self.size:
            parts[self.size-1]=self.split.join(parts[self.size:])
        rval = [self.line_processors[k] for k,part in enumerate(parts)]
        rval=sum(rval,[])
        return rval

class json_line_processing():

    def __init__(self, d_processors):
        self.d_processors = d_processors
        self.size=sum(x.size for x in d_processors.values())

    def __call__(self, line):
        rval=[]
        a=byteify(json.loads(line.strip()))
        for key in self.d_processors:
            process = self.d_processors[key]
            part=a[key]
            rval+=process(part)
        return rval

class sequence_label_line_processing():

    def __init__(self, words, tags, size=2, max_len=100, return_length=False):
        self.sq_process = sequence_line_processing( words , max_len = max_len, return_length=return_length )
        self.tag_process = label_line_processing( tags )
        self.size=size

    #@tryfunc()
    def __call__(self, line):
        words = line.strip().split('\t')
        tags=words[0]
        seq=' '.join(words[1:])
        #a,b = self.sq_process(seq), self.tag_process(tags)
        #print type(a),a.shape
        #print type(b),b.shape
        rval = self.sq_process(seq)+self.tag_process(tags)
        return rval

class LineBasedDataset():

    def __init__(self, data_path, line_processing=None, queue_size=100000, save_all=False, use_length=True,
            len=len, batch_size=100):
        self.file_reader = None
        self.data_path = data_path
        self.queue_size=queue_size
        self.line_processing=line_processing
        self.size=line_processing.size
        self.save_all=False
        self.len=len
        self.batch_size = batch_size
        if save_all:
            self.all_data=list(self.epoch_data())
            self.save_all=save_all
            self.len=len(self.all_data)
            print 'len===============',self.len
    
    def __len__(self):
        if self.len:
            return self.len
        else:
            self.len=len(list(self.epoch_data()))

    def __iter__(self):
        if self.save_all:
            for d in self.all_data:
                yield d
            return
        self.queue = Queue.Queue()
        print self.data_path
        self.file_reader = open(self.data_path, 'r')
        while True:
            if False:
            # if not self.queue.empty():
                #yield self.queue.get()
                pass
            else:
                batch=[[],]*self.size
                k=0
                for line in self.file_reader:
                    k+=1
                    items=self.line_processing(line)
                    if not items or len(items)!=self.size:
                        #print items
                        #print 'Not correct parsed:\t'
                        # print line,len(line)
                        # raw_input('xxxxxxxxxxxxx')
                        continue
                    for k,item in enumerate(items):
                        batch[k]=batch[k]+[item]
                    if len(batch[0])==self.batch_size:
                        batch=[np.array(x) for x in batch]
                        #self.queue.put( batch )
                        #print 'put'
                        yield batch
                        batch=[[],]*self.size
                    #if self.queue.qsize()==self.queue_size:
                    #    break
                if k==0:break

class MultiDataset():

    def __init__(self, datasets):
        self.datasets=datasets
        self.size=sum(d.size for d in datasets)

    def epoch_data(self):
        self.iterators=[x.epoch_data() for x in self.datasets]
        while True:
            batch_item = []
            for it in self.iterators:
                item=it.next()
                if type(item)==type([1,2]):
                    batch_item+=item
                else:
                    batch_item.append(item)
            if not batch_item:
                yield bctch_item
            else:
                break

    def __len__(self):
        return len(self.datasets[0])

class SampleDataset():

    def __init__(self, datasets, ratios):
        self.datasets=datasets
        self.ratios=[sum(ratios[:i+1])*1.0/sum(ratios) for i in range(len(ratios))]
        self.size=sum(d.size for d in datasets)

    def select(self,r):
        for i in range(len(self.ratios)):
            if self.ratios[i]>=r:
                return i
        
    def epoch_data(self):
        iterators=[x.epoch_data() for x in self.datasets]
        finished=[0,]*len(self.ratios)
        finished[1]=1
        # TODO 这里手动让第二数据集直接结束
        while True:
            r=random.random()
            k=self.select(r)
            batch_item=[None,]*self.size
            item=None
            try:
                item=iterators[k].next()
                mask=[0,]*len(self.datasets)
                mask[k]=1
                if not isinstance(item, list): item=[item]
                size=sum(d.size for d in self.datasets[:k])
                batch_item[size:size+len(item)]=item
                batch_item[size+len(item):size+2*len(item)]=item
                # 上面一行指导了无效数据
                batch_item.append(np.array(mask))
                yield batch_item
            except Exception,e:
                print e
                finished[k]=1
                if sum(finished)==len(finished):
                    break
    
    def __len__(self):
        self.len=sum(len(d) for d in  self.datasets)
        return self.len
