import tensorflow as tf
        
cpu_conf = tf.ConfigProto(
      device_count = {'CPU': 12, 'GPU':0}, 
      allow_soft_placement=True,
      log_device_placement=False,)
def cpu_sess():
    return tf.Session(config=cpu_conf)
