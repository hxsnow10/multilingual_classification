#encoding=utf-8
from collections import defaultdict
import os

def load_synonym(synonym_path, only_word=True, languages=None):
    if not os.path.isfile(synonym_path):
        return {}
    ii=open(synonym_path,'r')
    synonym=defaultdict(set)
    for line in ii: 
        try:
            line_,weight=line.strip().split('\t')
            weight=float(weight)
            line=line_
        except:
            weight=1
            pass
        if '/' not in line or not line.strip() or '(' in line :continue
        parts=line.strip().split('/')
        if only_word:
            parts=[part for part in parts if ' ' not in part]
        if languages is not None:
            parts=[part for part in parts if part.split('@')[0] in languages]
        if len(parts)<=1:continue
        for part in parts:
            synonym[part].update(parts)
    return synonym
