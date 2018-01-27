import os
import tensorflow as tf

def config_func():
    '''config depends on some parameters
    '''
    class config():
        # use for generate paths, not directly used
        branch='en'
        w2v_path='vecs_idf/vec.txt'
        idf_path='vecs_idf/en_de.idf'
        base_dir='./ted-cldc'
        tags_path='{}/tags.txt'.format(base_dir)
        input_model_path='model/model_en-188'
        output_model_path='model/model_new'

        # model 
        sen_len=45000
        vec_len=300
        text_repr='add+idf'
        filter_sizes=[1]
        filter_nums=[100]
        exclusive=False
    return config
