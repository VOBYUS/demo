#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

tf.compat.v1.reset_default_graph()
imported_graph = tf.train.import_meta_graph('./my_model1.meta')
sess = tf.Session()
imported_graph.restore(sess, './my_model1')
graph = tf.get_default_graph()


def how_drowsy(blinks):
    labels_net = np.array([10]).reshape([1,1])
    Placeholder = 1
    phase_train = False
    return sess.run(['last_fc/mul:0'],
                    feed_dict={
                     'bacth_in:0': blinks,
                     'labels_net:0': labels_net,
                     'Placeholder:0': Placeholder,
                     'phase_train:0': phase_train,
                     }
                    )


if __name__ == '__main__':
    Blinks = np.load('Blinks_30_Fold1.npy')
    Blinks[:,:,0] = (Blinks[:,:,0] - np.mean(Blinks[:,:,0])) / np.std(Blinks[:,:,0])
    Blinks[:,:,1] = (Blinks[:,:,1] - np.mean(Blinks[:,:,1])) / np.std(Blinks[:,:,1])
    Blinks[:,:,2] = (Blinks[:,:,2] - np.mean(Blinks[:,:,2])) / np.std(Blinks[:,:,2])
    Blinks[:,:,3] = (Blinks[:,:,3] - np.mean(Blinks[:,:,3])) / np.std(Blinks[:,:,3])
    bacth_in = Blinks
    print(how_drowsy(bacth_in))
