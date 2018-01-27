#!/usr/bin/env python
# encoding=utf-8
import tensorflow as tf
from nlp.word2vec import load_w2v
import sys
from nlp.synonym import load_synonym
from collections import OrderedDict
import os
import numpy as np
from nlp import get_lang
import argparse
#LANGS = sys.argv[4].split('-')

def create_emb_log(emb_file, words_file, log_dir, modes, topns):
    LOG_DIR = log_dir
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    w2v = load_w2v(emb_file)
    synonym = load_synonym(words_file)
    session_conf = tf.ConfigProto(
          device_count = {'CPU': 1, 'GPU':0}, 
          allow_soft_placement=True,
          log_device_placement=False,)
    with tf.Session(config=session_conf) as sess:
        from tensorflow.contrib.tensorboard.plugins import projector
        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter(LOG_DIR)
        for mode,topn in zip(modes, topns):
            meta_file_path = 'meta_data_{}_{}.tsv'.format(mode, topn)
            model_path = 'model_{}_{}'.format(mode,topn)
            emb_name = '{}_{}_{}'.format(emb_file,mode,topn)
            if mode:
                words_=set()
                for w in synonym:
                    if w not in w2v :continue
                    l1=get_lang(w)
                    num=0
                    for w_ in synonym[w]:
                        l2=get_lang(w_)
                        if w_ not in w2v or w==w_:continue
                        if 'strict' in mode and l2==l1:continue
                        words_.update([w,w_])
                        num+=1
                        if mode=='strict'and num>=5:break
                        if topn and len(words_)>=topn:break
                    if topn and len(words_)>=topn:break
            else:
                if topn:
                    words_=w2v.keys()[:topn]
                else:
                    words_=w2v.keys()
            w2v = OrderedDict([(w,w2v[w]) for w in w2v if w in words_])
            w2l = OrderedDict([(w,get_lang(w)) for w in w2v if w in words_])
            print '-'*40
            print "meta_file_path={}".format(meta_file_path)
            print "model_path={}".format(model_path)
            print "mode={}, topn={}".format(mode, topn)
            print "w2v size={}".format(len(w2v))
            meta_file=open(os.path.join(LOG_DIR, meta_file_path), 'w')
            meta_file.write('Name\tLang\n')
            for word in w2v:
                meta_file.write(word+'\t'+w2l[word]+'\n')
            meta_file.close()
                
            embedding_var = tf.Variable(np.array(w2v.values()), dtype=tf.float32, name=emb_name)
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(LOG_DIR, model_path))

            # Use the same LOG_DIR where you stored your checkpoint.

            # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto

            # You can add multiple embeddings. Here we add only one.
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = meta_file_path
            print embedding
            print embedding.tensor_name, embedding.metadata_path
            
            # Saves a configuration file that TensorBoard will read during startup.
            projector.visualize_embeddings(summary_writer, config)
            # 这个不能写到外面一层去

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path')
    parser.add_argument('--synonym_path', default='')
    parser.add_argument('--log_dir', default='')
    parser.add_argument('--synonym_modes', default='"strict"/"soft"/"strict"')
    parser.add_argument('--topns', default='None/300/300')
    #parser.add_argument('--langs', default='')
    args = parser.parse_args()
    print args
    args.synonym_modes = [eval(mode) for mode in args.synonym_modes.split('/')]
    args.topns = [eval(topn) for topn in args.topns.split('/')]
    print args
    create_emb_log(args.emb_path, args.synonym_path, args.log_dir, args.synonym_modes, args.topns)
