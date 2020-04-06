#将ckpt文件转化为pb文件，前提是需要知道网络的输出节点名称, 如果不指定输出节点名称, 程序就不知道该freeze哪些节点, 就没有办法保存模型.
import tensorflow as tf
from tensorflow.python.framework import graph_util
import  os
from tensorflow.python.platform import gfile

def ckpt_freeze_graph(ckpt, output_graph):
    # if os.path.isdir(output_graph) is False:
    #     os.makedirs(output_graph)
    #output_node_names = 'bert/encoder/layer_7/output/dense/kernel'
    # saver = tf.train.import_meta_graph(ckpt+'.meta', clear_devices=True)
    output_node_names = 'op_to_store'
    saver = tf.compat.v1.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(',')
        )
        with tf.gfile.GFile(output_graph, 'wb') as fw:
            fw.write(output_graph_def.SerializeToString())
        print('{} ops in the final graph.'.format(len(output_graph_def.node)))
ckpt = './ckpt/model.ckpt'
output_graph = './ckpt_to_pb/ckpt_to_pb.pb'
ckpt_freeze_graph(ckpt,output_graph)