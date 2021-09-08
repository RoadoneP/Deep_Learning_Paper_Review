import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib

w = np.random.randn(100, 20)
w_shape = [100, 20]
a = tf.reshape(w, [-1, w_shape[-1]])

u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
print(u)