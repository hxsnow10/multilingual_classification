# encoding=utf-8
#import nltk

def nltk_tok(line):
    '''以sents.py为例，他问题很大'''
    try:
        tokens = nltk.word_tokenize(line.strip())
        line2=' '.join(tokens)
    except:
        print 'error',line
        line2=line
    return line2

class SpiltMergeTokbyre():

    def __init__(self, splits, merge_rules=[]):
        escape_spilts=[re.escape(s).decode('utf-8') for s in splits]
        self.splits=re.compile(u'('+u'|'.join(escape_splits)+u')')
        self.merge_rules=merge_rules

    def cut(self, sent):
        sent=sent.decode('utf-8')
        toks = self.splits.split(sent)
        toks = self.merge(toks)
        toks = [tok.encode('utf-8') for tok in toks if tok!=' ']
        return toks

class SpiltMergeTok():

    def __init__(self, splits, merge_words=[], slower=False):
        self.splits=set(splits)
        self.merge_words=merge_words
        print 'tok inited'

    def cut(self, sent):
        sent=sent.lower()
        sent=sent.decode('utf-8')
        toks=[]
        tmp=u''
        for x in sent:
            if x in self.splits:
                toks.append(tmp)
                toks.append(x)
                tmp=u''
            else:
                tmp+=x
        toks.append(tmp)
        toks=[tok for tok in toks if tok!=u'']
        toks = self.merge(toks)
        toks = [tok.encode('utf-8') for tok in toks if tok!=' ']
        return toks

    def merge(self, toks):
        rval=[]
        k=-1
        while k+1<len(toks):
            k=k+1
            tok=toks[k]
            merged=False
            for words in self.merge_words:
                if words==toks[k:k+len(words)]:
                    s=''.join(words)
                    rval.append(s)
                    k=k+len(words)-1
                    merged=True
                    break
            if not merged:
                rval.append(tok)
        return rval
                
# refer to https://en.wikipedia.org/wiki/Punctuation
#
en_splits=u"""’'[]【】()（）{}《》⟨⟩:：,，،、‒–—―….!！.。‹›«»--?？‘’“”''"";；/⁄ """
ar_splits=u"""،؛؟"""
# fr splits has different meanings with en
end_marks=['.','。','?','？','!','！','؟']
merge_rules=[[u"doesn",u"'",u"t"],[u"don",u"'",u"t"]]

ft=foreign_tok=SpiltMergeTok(en_splits+ar_splits,merge_rules)

def tok(text):
    return ft.cut(text)

def ar_tok(text):
    sents=text.split('\n')
    tokss=[ft.cut(sent)[::-1] for sent in sents]
    toks=sum(tokss,[])
    return toks

from chineseSegment.segment import Segment
import json

seg1=Segment()
seg2=Segment(user_dict_path=['zh_wiki_dict.txt'])
def zh_wiki_tok(sent):
    return seg2.cut(sent)
def zh_tok(sent):
    return seg1.cut(sent)

def tok_(sent):
    return tok(sent)

if __name__=='__main__':
    s=" Fuck you. I say 'you are rubbish'" 
    import time
    start=time.time()
    for i in range(100):
        print ft.cut(s)
    print (time.time()-start)/10
