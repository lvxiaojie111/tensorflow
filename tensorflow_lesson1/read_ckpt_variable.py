
'''
说明：只会保存变量的名字，常量、占位符无法读出
输出结果如下：
tensor_name:  b
1

'''
import os
from tensorflow.python import pywrap_tensorflow
def get_ckpt_alltensor_1():
    savedir='./ckpt/'#路径问题，最后加反斜杠，如果不加会认为斜杠后是文件名，而不是路径名
    savefile="model.ckpt"
    checkpoint_path = os.path.join(savedir,savefile)
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
      print("tensor_name: ", key)
      print(reader.get_tensor(key))

get_ckpt_alltensor_1()

'''
说明：只会输出变量的名字：
输入结果：
debug_string:

'b (DT_INT32) []\n'
'''
import pprint
import tensorflow as tf
def get_ckpt_alltensor_2():
    savedir = './ckpt/'  # 路径问题，最后加反斜杠，如果不加会认为斜杠后是文件名，而不是路径名
    savefile = "model.ckpt"
    checkpoint_path = os.path.join(savedir, savefile)
    NewCheck = tf.train.NewCheckpointReader(checkpoint_path)
    print("debug_string:\n")
    pprint.pprint(NewCheck.debug_string().decode("utf-8"))
get_ckpt_alltensor_2()

'''
输出结果：
tensor_name:  b
1
# Total number of params: 1
'''
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# print_tensors_in_checkpoint_file(os.path.join(savedir,savefile),None,True)#打印模型文件
def get_ckpt_alltensor_3():
    savedir = './ckpt/'  # 路径问题，最后加反斜杠，如果不加会认为斜杠后是文件名，而不是路径名
    savefile = "model.ckpt"
    print_tensors_in_checkpoint_file(os.path.join(savedir,savefile), #ckpt文件名字
                     None, # 如果为None,则默认为ckpt里的所有变量
                     True, # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
                     True) # bool 是否打印所有的tensor的name
get_ckpt_alltensor_3()
