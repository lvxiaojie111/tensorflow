import tensorflow as tf
import os


def save_model_ckpt(ckpt_file_path):
    # if not ckpt_file_path:
    #     os.makedirs(ckpt_file_path)
    output_graph=True
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    xy = tf.multiply(x, y)
    op = tf.add(xy, b, name='op_to_store')

    sess = tf.Session()
    if output_graph:
        tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())

    path = os.path.dirname(os.path.abspath(ckpt_file_path))
    if os.path.isdir(path) is False:
        os.makedirs(path)

    tf.train.Saver().save(sess, ckpt_file_path)

    # test
    feed_dict = {x: 2, y: 3}
    print(sess.run(op, feed_dict))
    if output_graph:
        tf.summary.FileWriter("logs/",sess.graph)
save_model_ckpt("./ckpt/model.ckpt")

