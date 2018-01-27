# encoding=utf-8
import os

cur_dir = os.path.dirname( os.path.abspath(__file__)) or os.getcwd()
code2lang={}
ii=open(os.path.join(cur_dir, 'languages.txt'), 'r')
for line in ii:
    try:
        en_lang, local_lang, code = line.strip().split('\t')
        code2lang[code]=[en_lang, local_lang]
    except:
        print len(line.strip().split('\t'))
wikipedia_langs=code2lang
