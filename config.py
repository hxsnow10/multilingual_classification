import os
import tensorflow as tf

def config_func(target_, source_='en'):
    '''config depends on some parameters
    '''
    class config():
        # use for generate paths, not directly used
        source=source_
        target=target_
        f=source+'-'+target
        b=target+'-'+source
        base_dir = './ted-cldc'
        
        # input paths
        # share en & tgt
        share_emb=True
        limited_words=True
        # source_w2v_path = target_w2v_path = 'vecs_idf/vec.txt'
        target_w2v_path = source_w2v_path = '/opt/xia.hong/data/word_vectors/concept-numberbatch/cpt_vec_{}_{}.txt'.format(source,target)
        source_w2v_path = target_w2v_path = '/opt/xia.hong/data/wiki_data/wikipedia_data/de/de_aligned_vec_merged.txt'
        source_w2v_path = target_w2v_path = '/opt/xia.hong/data/wiki_data/wikipedia_data/de/de_online_synonym_cpt.vec'
        
        # not share
        # source_w2v_path = '/opt/xia.hong/data/word_vectors/concept-numberbatch/cpt_vec_{}.txt'.format(source)
        # target_w2v_path = '/opt/xia.hong/data/word_vectors/concept-numberbatch/cpt_vec_{}.txt'.format(target)
        source_idf_path = 'vecs_idf/en_de.idf'
        target_idf_path = source_idf_path # share with source
        tags_path='{}/tags2.txt'.format(base_dir)
        
        train_data_path='{}/{}/train.txt'.format(base_dir ,f)
        dev_data_path='{}/{}/test.txt'.format(base_dir , f)
        test_data_paths = [
            '{}/{}/train.txt'.format(base_dir, b),
            '{}/{}/test.txt'.format(base_dir, b)]
        
        # output
        model_dir='./model2'
        model_path=os.path.join(model_dir,'model')
        # here source_model=target_model except emb
        summary_dir='./log_synonym_addidfwith0'
        
        # model
        max_vocab_size=None
        sen_len=45000
        vec_len=300
        text_repr='add+idf'
        filter_sizes=[1]
        filter_nums=[100]
        dropout_ratio=0.7
        exclusive=False
        l2_lambda=0
        learning_rate=0.005
        
        # other
        ask_for_del=False
        epoch_num=120
        batch_size=5
        summary_steps=5
        session_conf = tf.ConfigProto(
              device_count = {'CPU': 12, 'GPU':0}, 
              allow_soft_placement=True,
              log_device_placement=False,)
        session_conf=None 
    return config
