import os
import os.path as op
import numpy as np
from configs import globals as gb
from inputs.input import Dataset
import tensorflow as tf
from importlib import import_module

from_detection_checkpoint: True

"""
load global variables
"""
ROOT_DIR=gb.ROOT_DIR
DATASET=gb.DATASET
MODALITY=gb.MODALITY
PROJMETHOD=gb.PROJMETHOD
NUMB_CHANNELS=gb.NUMB_CHANNELS
INPUT_IMAGESIZE = gb.INPUT_IMAGESIZE
"""
define the train() function
"""
def train(dataset_train, ckptfile="", otsmodel=""):
    is_finetune = bool(ckptfile)

    start_step=0 if not is_finetune else int(ckptfile.split('-')[-1])
    ## placeholders for graph input
    views_ = tf.placeholder('float32', shape=(None, NUMB_CHANNELS, INPUT_IMAGESIZE[0], INPUT_IMAGESIZE[1], 3), name='input')
    labels_ = tf.placeholder('int32', shape=(None), name='label')
    model_name =import_module('model.'+gb.BASE_NETWORK)
    shapenet = model_name.resnet50(views_,is_training=True)
    model_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES)
    init_op = tf.initialize_all_variables()
    path_saver = "/home/jiaxinchen/Project/3DXRetrieval/code/proposed/model/resnet_v1_50.ckpt"
    saver = tf.train.Saver(model_variables)

    global_step = tf.Variable(start_step, trainable=False)
    step=start_step
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore( sess,path_saver)
        xxx=dataset_train._batches(gb.BATCH_SIZE)
        for batch_x, batch_y in xxx:#dataset_train.batches(gb.BATCH_SIZE):
            step = step+1
            print("Go step %d"%(step))
            val_feed_dict = {views_:  batch_x,
                             labels_: batch_y}
            xx=sess.run( shapenet, feed_dict = val_feed_dict)
    print(0)



"""
define the main() function
"""
def main():
    train_lists_dir  = op.join(ROOT_DIR, "data", DATASET, DATASET+"_"+MODALITY+"_"+PROJMETHOD+"_Train_Lists.txt" )
    listfiles, labels = read_lists( train_lists_dir )
    dataset_train = Dataset(listfiles, labels)
    if gb.FLAG_SHUFFLE:
        dataset_train.shuffle()
    train(dataset_train)

def read_lists( lists_dir ):
    listfile_labels= np.loadtxt(lists_dir, dtype=str).tolist()
    listfile_labels.sort()
    listfiles, labels = zip(*[ ( ROOT_DIR+l[0], int(l[1]) ) for l in listfile_labels])
    listfiles_full = list([])
    labels_full = list([])
    for i in range(len(listfiles)):
        with open( listfiles[i], 'r' ) as f:
            texts = f.readlines()
            for j in range(4, len(texts), NUMB_CHANNELS+1):
                listfiles_full.append(texts[j:j+NUMB_CHANNELS])
                labels_full.append(labels[i])

    return listfiles_full, labels_full


if __name__=='__main__':
    main()