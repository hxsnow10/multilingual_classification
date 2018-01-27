# encoding=utf-8
'''give a example : modelA(emb_1) ---> modelA(emb2)
there are two directions:
* use code which defines modelA
* don't use model define code
'''

from text_cnn import TextClassifier
from text_cnn import *
from text_cnn import config
from dynamic_config import config_func

def save_model_with_w2v():
    tags,_ = get_vocab(config.tags_path)
    w2v = load_w2v(config.w2v_path)
    init_embedding = np.array(w2v.values(), dtype=np.float32)
    if config.text_repr=='add+idf':
        idf=load_w2v(config.idf_path)
        init_idf = np.array([float(idf.get(w,10.0)) for w in w2v], dtype=np.float32)
    with tf.Session() as sess:
        model = TextClassifier(
            num_classes=len(tags), 
            init_embedding=init_embedding, 
            init_idf=init_idf, 
            emb_name=config.branch+'_emb',
            reuse=False,
            mode='eval',
            name_scope=config.branch,
            configp=config)
        print "INPUTS:\n\t", '\n\t'.join(str(x) for x in model.inputs)
        print "OUTPUTS", model.predictions
        sess.run(model.init)
        model.train_saver.restore(sess, config.input_model_path)
        model.all_saver.save(sess,config.output_model_path)
        # here we can do some test about the new model
        
if __name__=="__main__":
    global config
    config=config_func()
    save_model_with_w2v()
