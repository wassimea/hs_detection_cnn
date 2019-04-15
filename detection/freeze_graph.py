import tensorflow as tf
import numpy as np
import cv2
import sys
#from train_models.mtcnn_model import O_Net
import os
import chl
import time
from tensorflow.python.platform import gfile


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    #checkpoint = tf.train.get_checkpoint_state(model_dir)
    #input_checkpoint = checkpoint.model_checkpoint_path
    input_checkpoint = '/home/wassimea/Desktop/wassimea/work/train_models/mn/3_mobilenet_mse/-2'
    
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = "/home/wassimea/Desktop/frozen_model_mn_chl.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 
        for node in output_graph_def.node:
        	if node.op == 'RefSwitch':
        	  node.op = 'Switch'
        	  for index in xrange(len(node.input)):
        	    if 'moving_' in node.input[index]:
        	      node.input[index] = node.input[index] + '/read'
        	elif node.op == 'AssignSub':
        	  node.op = 'Sub'
        	  if 'use_locking' in node.attr: del node.attr['use_locking']
        	elif node.op == 'AssignAdd':
        	  node.op = 'Add'
        	  if 'use_locking' in node.attr: del node.attr['use_locking']
        	elif node.op == 'Assign':
        	  node.op = 'Identity'
        	  if 'use_locking' in node.attr: del node.attr['use_locking']
        	  if 'validate_shape' in node.attr: del node.attr['validate_shape']
        	  if len(node.input) == 2:
        	    # input0: ref: Should be from a Variable node. May be uninitialized.
        	    # input1: value: The value to be assigned to the variable.
        	    node.input[0] = node.input[1]
        	    del node.input[1]

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


def freeze_graph_good():

    model_path="/home/wassimea/Desktop/smats_cls/model.pb"
    # read graph definition
    f = gfile.FastGFile(model_path, "rb")
    gd = graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    # fix nodes

    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if('moving_' in node.input[index]):
                    nod.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: 
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                    del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            if 'validate_shape' in node.attr:
                    del node.attr['validate_shape']
            if len(node.input) == 2:
                node.input[0] = node.input[1]
                del node.input[1]
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    # import graph into session
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, './', '//home/wassimea/Desktop/smats_cls/good_frozen.pb', as_text=False)


def main(_):
    model_dir = '/home/wassimea/Desktop/wassimea/work/train_models/mn/3_mobilenet_mse/'
    output_node_names = 'save/restore_all'
    #freeze_graph(model_dir, output_node_names)
    freeze_graph_good()


if __name__ == '__main__':
  tf.app.run()
