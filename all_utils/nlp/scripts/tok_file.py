#!/usr/bin/env python
# encoding=utf-8
#from nlp import tokenize 
import json
import argparse

from nlp.tokenize import tok, zh_tok
from nlp import is_zh
from chineseSegment.segment import Segment
from utils.mpwork import BatchedMpWork,reader,writer
from utils import byteify

class processor():
    def __init__(self, tok):
        self.tok=tok

    def __call__(self, line):
        toks=self.tok(line.strip())
        rval=' '.join(toks)+'\n'
        return rval

def main(input_path, output_path, tok_mode):
    if tok_mode=='zh':
        process=processor(zh_tok)
    else:
        process=processor(tok)
    BatchedMpWork(data=reader(input_path), process=process, listener=writer(output_path), workers=10)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_path')
    parser.add_argument('-o','--output_path')
    parser.add_argument('-m','--tok_mode')
    args = parser.parse_args() 
    main(args.input_path, args.output_path, args.tok_mode)
