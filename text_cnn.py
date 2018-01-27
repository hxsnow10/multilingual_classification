# encoding=utf-8
import sys
import os
from os import makedirs
from shutil import rmtree
import json

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score
from nlp import load_w2v
from nlp.base import get_words
from tf_utils.data import sequence_label_line_processing, LineBasedDataset
from tf_utils.model import multi_filter_sizes_cnn 
from utils.base import get_vocab
from utils.base import get_func_args

from config import config_func

config=None

class TextClassifier(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self,
            num_classes,
            vocab_size=None,
            init_embedding=None,
            init_idf=None,
            class_weights=None,
            emb_name=None,
            reuse=None,
            mode='train',
            name_scope=None,
            configp=None):
        # 为了dynamic_model时候, 那里的config能替换这个函数的config
        # 如果不加这一段，dynamic_model中全局config被更新了，这里的config还是None
        if configp:
            global config
            config=configp
        
        args=get_func_args()
        for arg in args:
            setattr(self, arg, args[arg])
        
        with tf.name_scope(self.name_scope):
            self.build_inputs()
            self.build_embeddings()
            if config.text_repr=='cnn':
                self.build_sent_cnn()
            elif config.text_repr=='add':
                self.build_sent_add()
            elif config.text_repr=='add+idf':
                self.build_sent_add_idf()
            if config.exclusive:
                self.build_exclusive_ouputs(self.sent_vec, self.num_classes)
            else:
                self.build_nonexclusive_outputs(self.sent_vec, self.num_classes)
            
            if mode=='train':
                self.step_summaries = tf.summary.merge_all()         
                self.train_op = tf.train.AdamOptimizer(config.learning_rate, name="adam_{}".format(emb_name)).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.all_vars=list(set(
            (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)+
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="share")+
            [x for x in tf.global_variables() if x.name==self.emb_name+":0"])))
        self.train_vars=[x for x in self.all_vars if x in tf.trainable_variables()]
        self.all_saver=tf.train.Saver(self.all_vars)
        self.train_saver = tf.train.Saver(self.train_vars)
        print 'ALL VAR:\n\t', '\n\t'.join(str(x) for x in self.all_saver._var_list)
        print 'TRAIN VAR:\n\t', '\n\t'.join(str(x) for x in self.train_saver._var_list)
        print 'INPUTS:\n\t', '\n\t'.join(str(x) for x in self.inputs)
        print 'OUTPUTS:\n\t', '\n\t'.join(str(x) for x in self.outputs)

        
    def build_inputs(self):
        # Placeholders for input, output and dropout
        print id(config)
        self.input_x = tf.placeholder(tf.int64, [None, config.sen_len], name="input_x")
        self.input_sequence_length = tf.placeholder(tf.int64, [None], name="input_l")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        if self.mode=='train':
            self.input_y = tf.placeholder(tf.int64, [None, self.num_classes], name="input_y")
            self.class_weights=tf.constant(self.class_weights)
            self.inputs = [ self.input_x, self.input_sequence_length, self.input_y, self.dropout_keep_prob ]
        else:
            self.inputs = [ self.input_x, self.input_sequence_length, self.dropout_keep_prob ]
        self.batch_size=tf.shape(self.input_x)[0]
        self.outputs = []
        if config.text_repr=='add+idf':
            self.idf = tf.Variable(self.init_idf, dtype=tf.float32, name='idf', trainable=False)

    def build_embeddings(self):
        # Embedding layer
        # tf not allowed init value of any tensor >2G
        init_emb=tf.constant(self.init_embedding, dtype=tf.float16)
        if self.init_embedding is not None:
            W = tf.get_variable(self.emb_name, initializer=init_emb, trainable=False)
        else:
            pass
        self.words_emb = tf.cast(tf.nn.embedding_lookup(W, self.input_x), tf.float32)
        self.words_emb = tf.expand_dims(self.words_emb, -1)

    def build_sent_cnn(self):
        with tf.variable_scope("share"):
            self.sent_vec = multi_filter_sizes_cnn(self.words_emb, config.sen_len, config.vec_len, config.filter_sizes, config.filter_nums, name='cnn', reuse=self.reuse)
        with tf.name_scope("dropout"):
            self.sent_vec = tf.nn.dropout(self.sent_vec, self.dropout_keep_prob)
        
    def build_sent_add(self):
        self.mask = tf.cast(tf.sequence_mask(self.input_sequence_length, config.sen_len), tf.float32)
        self.sent_vec = tf.reduce_sum(tf.expand_dims(self.mask,-1)*self.words_emb[:,:,:,0],1)
        self.sent_vec = tf.nn.l2_normalize(self.sent_vec, dim = -1)
        
    def build_sent_add_idf(self):
        self.mask = tf.cast(tf.sequence_mask(self.input_sequence_length, config.sen_len), tf.float32)
        self.idf_x = tf.nn.embedding_lookup(self.idf, self.input_x)
        self.sent_vec = tf.reduce_sum(tf.expand_dims(self.mask,-1)*self.words_emb[:,:,:,0]\
                    *tf.expand_dims(self.idf_x,-1),1)
        self.sent_vec = tf.nn.l2_normalize(self.sent_vec, dim = -1)
        
    def build_exclusive_ouputs(self, inputs, num_classes):
        with tf.name_scope("output"):
            with tf.variable_scope("share"):
                self.scores = tf.layers.dense(inputs, num_classes, name="dense", reuse=self.reuse)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.outputs.append(self.predictions)
        if self.mode!='train':return

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.l2_loss=0#TODO
            self.loss = tf.reduce_mean(losses) + config.l2_lambda * self.l2_loss
        tf.summary.scalar("loss", self.loss)    
        self.outputs.append(self.loss)
        
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        # tf.summary.scalar("accuracy", self.accuracy)    

    def build_nonexclusive_outputs(self, inputs, num_classes):
        with tf.name_scope("output"):
            with tf.variable_scope("share"):
                self.scores = tf.layers.dense(inputs, num_classes*2, name="dense", reuse=self.reuse)
            self.scores = tf.reshape(self.scores, [self.batch_size, num_classes, 2])
            self.predictions = tf.argmax(self.scores, -1, name="predictions")
            self.outputs.append(self.predictions)
        if self.mode!='train':return

        with tf.name_scope("loss"):
            input_y = tf.one_hot(self.input_y,depth=2,axis=-1)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=input_y)
            self.loss = tf.reduce_mean(losses*tf.expand_dims(self.class_weights,0))
        tf.summary.scalar("loss", self.loss)    
        self.outputs.append(self.loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.class_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), axis = 0, name="class_accuracy")
        # tf.summary.scalar("accuracy", self.accuracy)    
            

def train(sess, source_model, target_model, 
        train_data, dev_data, test_datas, tags,
        summary_writers, class_summary_writers):
    sess.run(source_model.init)
    sess.run(target_model.init)
    step=0
    best_score=0
    for batch in range(config.epoch_num):
        for k,inputs in enumerate(train_data):
            step+=1
            fd=dict(zip(source_model.inputs, inputs+[config.dropout_ratio]))
            if step%config.summary_steps!=0:
                loss,_=sess.run([source_model.loss, source_model.train_op], feed_dict=fd)
            else:
                loss,_,summary=\
                    sess.run([source_model.loss, source_model.train_op, source_model.step_summaries], feed_dict=fd)
                summary_writers['train'].add_summary(summary, step)
            print "batch={}\tstep={}\tglobal_step={}\tloss={}".format(batch, k, step ,loss)
            # eval every batch
            if k==0 and batch>=1:
                _,train_data_metrics = evaluate(sess,source_model,train_data,tags)
                score,dev_data_metrics = evaluate(sess,source_model,dev_data,tags)
                test_data_metricss = [evaluate(sess,target_model,test_data,tags)[1]
                    for test_data in test_datas]
                def add_summary(writer, metric, step):
                    for name,value in metric.iteritems():
                        summary = tf.Summary(value=[                         
                            tf.Summary.Value(tag=name, simple_value=value),   
                            ])
                        writer.add_summary(summary, global_step=step)
                add_summary(summary_writers['train'], train_data_metrics, step)
                add_summary(summary_writers['dev'], dev_data_metrics, step)
                add_summary(summary_writers['test-1'], test_data_metricss[0], step)
                add_summary(summary_writers['test-2'], test_data_metricss[1], step)
                
                if score>best_score:
                    best_score=score
                    # source_model.train_saver.save(sess, config.model_path, global_step=step)
                    source_model.train_saver.save(sess, config.model_path+'_'+source_model.name_scope, 
                            global_step=step)
                    target_model.train_saver.save(sess, config.model_path+'_'+target_model.name_scope, 
                            global_step=step)
                    source_model.all_saver.save(sess, config.model_path+'_'+source_model.name_scope+'_all')
                    target_model.all_saver.save(sess, config.model_path+'_'+target_model.name_scope+'_all')

def evaluate(sess, model, eval_data, target_names=None,restore=False):
    #model.saver.restore(sess, config.model_output)
    if restore:
        print 'reload model...'
        model.saver.restore(sess, config.model_output)
    total_y,total_predict_y = [], []
    print 'start evaluate...'
    for inputs in eval_data:
        fd=dict(zip(model.inputs, inputs+[1]))
        predict_y=\
            sess.run(model.predictions, feed_dict = fd)
        total_y = total_y + [inputs[2]]
        total_predict_y = total_predict_y + [predict_y]
    total_y = np.concatenate(total_y,0)
    total_predict_y=np.concatenate(total_predict_y,0)
    # print total_y.shape, total_predict_y.shape
    # print classification_report(total_y, total_predict_y, target_names=target_names)
    p,r,f=precision_score(total_y, total_predict_y, average='weighted'),\
        recall_score(total_y, total_predict_y, average='weighted'),\
        f1_score(total_y, total_predict_y, average='weighted')
    print p,r,f
    return f,{"precision":p,"recall":r,"f1":f}

def get_label_nums(train_data, tags=None):
    label_num=np.array([0,]*len(tags),dtype=np.float32)
    for sequence, _, labels in train_data:
        labels=np.sum(labels,axis=0)
        label_num+=labels
    return label_num

def main():
    def check_dir(dir_path, ask_for_del):
        if os.path.exists(dir_path):
            y=''
            if ask_for_del:
                y=raw_input('new empty {}? y/n:'.format(dir_path))
            if y.strip()=='y' or not ask_for_del:
                rmtree(dir_path)
            else:
                print('use a clean summary_dir')
                quit()
        makedirs(dir_path)
        oo=open(os.path.join(dir_path,'config.txt'),'w')
        d={}
        for name in dir(config):
            if '__' in name:continue
            d[name]=getattr(config,name)
        oo.write(json.dumps(d,ensure_ascii=False))
    check_dir(config.summary_dir, config.ask_for_del)
    check_dir(config.model_dir, config.ask_for_del)
    # assuse train and dev must use one emb
    # train and test may with differnent emb
    # for more info, refer to README
    if config.limited_words:
        limited_words=get_words([config.train_data_path, config.dev_data_path]+config.test_data_paths)
        limited_words=limited_words.keys()
    else:
        limited_words=[]
    tags,_ = get_vocab(config.tags_path)
    source_init_idf = target_init_idf = None
    if config.share_emb:
        assert config.target_w2v_path==config.source_w2v_path
        assert config.target_idf_path==config.source_idf_path
        w2v=load_w2v(config.source_w2v_path, max_vocab_size=config.max_vocab_size, limited_words=limited_words, norm=True)
        source_init_embedding=target_init_embedding=np.array(w2v.values(), dtype=np.float32)
        if config.text_repr=='add+idf':
            w2idf = load_w2v(config.source_idf_path)
            def drop(x):
                if x>=3:return x
                else:return 0
            source_init_idf = target_init_idf =\
                np.array([drop(float(w2idf.get(w,7.0))) for w in w2v], dtype=np.float32)
        words={k:w for k,w in enumerate(w2v.keys())}
        source_line_processing = target_line_processing =\
            sequence_label_line_processing(words, tags, max_len = config.sen_len, size=3, return_length=True)
    else:
        source_w2v=load_w2v(config.source_w2v_path, max_vocab_size=config.max_vocab_size, limited_words=limited_words, norm=True)
        target_w2v=load_w2v(config.source_w2v_path, max_vocab_size=config.max_vocab_size, limited_words=limited_words, norm=True)
        source_init_embedding = np.array(source_w2v.values(), dtype=np.float32)
        target_init_embedding = np.array(source_w2v.values(), dtype=np.float32)
        if config.text_repr=='add+idf':
            source_w2idf=load_w2v(config.source_idf_path)
            target_w2idf=load_w2v(config.source_idf_path)
            source_init_idf=np.array([float(source_w2idf.get(w,10.0)) for w in source_w2v], dtype=np.float32)
            target_init_idf=np.array([float(target_w2idf.get(w,10.0)) for w in target_w2v], dtype=np.float32)
        source_words = {k:w for k,w in enumerate(source_w2v.keys())}
        target_words = {k:w for k,w in enumerate(target_w2v.keys())}
        source_line_processing=\
            sequence_label_line_processing(source_words, tags, max_len = config.sen_len, size=3, return_length=True)
        target_line_processing=\
            sequence_label_line_processing(target_words, tags, max_len = config.sen_len, size=3, return_length=True)
   
    # datas
    train_data = LineBasedDataset(config.train_data_path, source_line_processing, batch_size= config.batch_size)
    
    dev_data = LineBasedDataset(config.dev_data_path, source_line_processing, batch_size = config.batch_size)
    test_datas = [LineBasedDataset(path, target_line_processing, batch_size = config.batch_size)
        for path in config.test_data_paths]
    
    # show shape
    for k,inputs in enumerate(train_data):
        print '-'*20,'batch ',k,'-'*20
        for inp in inputs:
            print inp.shape
        if k>=3:break
    
    # compute class weights for class unbalanced
    class_nums=get_label_nums(train_data, tags)
    class_weights=class_nums/np.sum(class_nums)*len(class_nums)
    print 'TRAIN CLASSES=\t',tags.values()
    print 'TRAIN CLASS_NUMS=\t',class_nums
    print 'TRAIN CLASS_WEIGHTS=\t',class_weights
     
    with tf.Session(config=config.session_conf) as sess:
        # use tf.name_scope to manager variable_names
        source_model=TextClassifier(
            num_classes=len(tags), 
            init_embedding=source_init_embedding, 
            init_idf=source_init_idf, 
            class_weights=class_weights,
            emb_name=config.source+'_emb',
            reuse=False,
            mode='train',
            name_scope=config.source)
        if config.share_emb:
            target_model=source_model
        else:
            target_model=TextClassifier(
                num_classes=len(tags), 
                init_embedding=target_init_embedding, 
                init_idf=target_init_idf, 
                class_weights=class_weights,
                emb_name=config.target+'_emb',
                reuse=True,
                mode='eval',
                name_scope=config.target)

        # summary writers for diiferent branch/class
        summary_writers = {
            sub_path:tf.summary.FileWriter(os.path.join(config.summary_dir,sub_path), flush_secs=5)
                for sub_path in ['train','dev','test-1','test-2']}
        class_summary_writers = {
            sub_path:
                {class_name:tf.summary.FileWriter(os.path.join(config.summary_dir,sub_path,class_name), flush_secs=5)
                    for class_name in tags.values()}
                for sub_path in ['train','dev','test-1','test-2']}
        
        # train source
        train(sess, source_model, target_model,
                train_data, dev_data, test_datas, 
                tags=tags.values(),
                summary_writers=summary_writers,
                class_summary_writers=class_summary_writers)

if __name__=='__main__':
    global config
    config = config_func('de')
    main()
