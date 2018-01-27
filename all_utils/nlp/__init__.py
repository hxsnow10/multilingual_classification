# encoding=utf-8
# from ltp import Segmentor, Postagger, Dpparser

from nlp.word2vec import load_w2v, save_w2v
from nlp.synonym import load_synonym
from polyglot.text import Text

import re
iszh=re.compile(u'[\u4e00-\u9fff]+', re.UNICODE)
isen=re.compile('[a-z_ ]+')
def is_zh(x):
    return iszh.search(x.decode('utf-8'))

def get_lang(text):
    if is_zh(text):return 'zh'
    if isen.match(text):return 'en'
    text_obj=Text(text)
    lang = text_obj.language.code
    return lang

from opencc import OpenCC
t2s_model=OpenCC()
def t2s(x):
    return t2s_model.convert(x).encode('utf-8')
