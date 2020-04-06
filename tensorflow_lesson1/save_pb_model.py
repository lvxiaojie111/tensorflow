import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

def save_mode_pb(pb_file_path):

  x = tf.placeholder(tf.int32, name='x')
  y = tf.placeholder(tf.int32, name='y')
  b = tf.Variable(1, name='b')
  xy = tf.multiply(x, y)
  # 这里的输出需要加上name属性
  op = tf.add(xy, b, name='op_to_store')
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  path = os.path.dirname(os.path.abspath(pb_file_path))

  if os.path.isdir(path) is False:

    os.makedirs(path)


  # convert_variables_to_constants 需要指定output_node_names，list()，可以多个

  constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

  with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:

    f.write(constant_graph.SerializeToString())


  # test

  feed_dict = {x: 2, y: 3}

  print(sess.run(op, feed_dict))
save_mode_pb("./pb/mypb.pb")