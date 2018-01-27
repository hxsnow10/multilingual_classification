# encoding=utf-8
import numpy as np
import tensorflow as tf
from tf_utils.data import sequence_line_processing
from tf_utils.predict import TFModel
'''
copy from 
INPUTS:
    Tensor("zh/input_x:0", shape=(?, 45000), dtype=int64)
    Tensor("zh/input_l:0", shape=(?,), dtype=int64)
    Tensor("zh/dropout_keep_prob:0", dtype=float32)
OUTPUTS Tensor("zh/output/predictions:0", shape=(?, 14), dtype=int64)
'''
class Model():
    def __init__(self, words_path, model_path):
        words={k:line.strip().split()[0] for k,line in enumerate(open(words_path).readlines()[:1])}
        self.line_processing=sequence_line_processing(words, max_len = 45000, return_length=True)
        session_conf = tf.ConfigProto(
            device_count = {'CPU': 12, 'GPU':0}, 
            allow_soft_placement=True,
            log_device_placement=False,)
        session_conf=None
        self.tf_sess=tf.Session(config=session_conf)
        self.tf_model=TFModel(
                self.tf_sess,
                model_path, 
                input_names=["en/input_x:0","en/input_l:0","en/dropout_keep_prob:0"], 
                output_names=["en/output/predictions:0"])

    def predict(self, text):
        input_x, input_l=self.line_processing(text)
        input_x=np.array([input_x],dtype=np.float32)
        input_l=np.array([input_l],dtype=np.float32)
        dropout_keep_prob=1
        data=[input_x, input_l, dropout_keep_prob]
        print data
        rval=self.tf_model.predict(data)
        return rval

if __name__=="__main__":
    #model_path='cnn_model/model_en_all'
    model_path='result/model1'
    model=Model('../dictionary_based/retrofit/model/new_vec2.txt',model_path)
    text='i like you'
    print model.predict(text)
