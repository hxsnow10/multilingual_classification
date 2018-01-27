#encoding=utf-8
from opencc import OpenCC

import time
class Tsconverter():
    def __init__(self):
        self.model={}
        for mode in ['t2s', 't2hk', 't2tw', 'tw2s', 'tw2sp','s2t', 's2hk', 's2tw', 's2twp']:
            self.model[mode]=OpenCC(mode+'.json')

    def convert(self, text, mode='t2s'):
        '''mode could be 
            't2s', 't2hk', 't2tw', 'tw2s', 'tw2sp'
            's2t', 's2hk', 's2tw', 's2twp'
        '''
        now=time.time()
        rval=self.model[mode].convert(text).encode('utf-8')
        # print time.time()-now
        return rval

tsmodel=Tsconverter()
def t2s(text):
    return tsmodel.convert(text, mode='t2s')
if __name__=='__main__':
    print t2s('纱布')
    print t2s('嚴格區分「一簡對多繁」和「一簡對多異」。')
    print t2s('engliah ')
