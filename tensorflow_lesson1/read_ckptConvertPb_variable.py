from tensorflow.python.platform import gfile
import os
import tensorflow as  tf
# 下载的谷歌训练好的Inception-v3模型文件目录
MODEL_DIR = './ckpt_to_pb/'
# Inception-v3模型文件名
MODEL_FILE = 'ckpt_to_pb.pb'
# 读取训练好的Inception-v3模型
# 二进制读取模型文件
# 读取并创建一个图graph来存放Google训练好的Inception_v3模型（函数）
def create_graph():
  with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
    # 新建GraphDef文件，用于临时载入模型中的图
    graph_def = tf.GraphDef()
    # GraphDef加载模型中的图
    graph_def.ParseFromString(f.read())
    # 在空白图中加载GraphDef中的图
    tf.import_graph_def(graph_def, name='')
    #print(reader.get_tensor(key))
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]#
    result_file = os.path.join(MODEL_DIR, 'ckpt_to_pb模型里面的张量名字result.txt')
    with open(result_file, 'w+') as f:
        for tensor_name in tensor_name_list:
            f.write(tensor_name+'\n')
# 创建graph
create_graph()
import tensorflow as tf
MODEL_DIR = './ckpt_to_pb/'
# Inception-v3模型文件名
MODEL_FILE = 'ckpt_to_pb.pb'
def creat_graph_operation():
  with tf.gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE),'rb')as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def,name='')
      with tf.Session()as sess:
          op_list = sess.graph.get_operations()
          result_file = os.path.join(MODEL_DIR, 'ckpt_to_pb模型里面的张量属性.txt')
          with open(result_file,'w+')as f:
              for index,op in enumerate(op_list):
                  f.write(str(op.name)+"\n")                   #张量的名称
                  f.write(str(op.values())+"\n")              #张量的属性
creat_graph_operation()