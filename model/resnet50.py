import tensorflow as tf
import configs.globals as gb

from model.nets.resnet_v1 import resnet_v1_50, resnet_arg_scope

_RGB_MEAN = [123.68, 116.78, 103.94]


def _view_pool(view_features, name):
    vp = tf.expand_dims(view_features[0], 0)
    for v in view_features[1:]:
        v=tf.expand_dims(v, 0)
        vp=tf.concat([vp,v], 0, name=name)
    vp = tf.reduce_max(vp, [0], name=name)
    return vp



def resnet50(views, is_training):
    if views.get_shape().ndims==4:
        print(0)
    elif views.get_shape().ndims==5:
        numb_chans = views.get_shape().as_list()[1]
        if( numb_chans!=gb.NUMB_CHANNELS ):
            raise ValueError("Number of channels of input view data does not match with the configured one")
        # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
        views = tf.transpose(views, perm=[1, 0, 2, 3, 4])
        views_pool = []
        for i in range(numb_chans):
            is_reuse = (i!=0)
            view=tf.gather(views,i)
            with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
                net, endpoints = resnet_v1_50(view, is_training=is_training, global_pool=True, reuse=is_reuse)
            xx=endpoints['resnet_v1_50/block4']
            views_pool.append(tf.reduce_mean(endpoints['resnet_v1_50/block4'], [1,2], name='pool5', keepdims=False))
        pool5_vp = _view_pool(views_pool, 'pool5_vp')
        
        return pool5_vp
    else:
        raise ValueError('Input must be of size [batch, height, width, 3] or [numb_channels, batch, height, width, 3]')