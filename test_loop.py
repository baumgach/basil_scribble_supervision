
# Simple loop for displaying predictions for random slices from the test dataset
#
# Usage:
#
# python test_loop.py path/to/experiment_logs
#
#
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse

import config.system as sys_config
import utils
import acdc_data
import image_utils
import model

def main(exp_config):

    # Load data
    data = acdc_data.load_and_maybe_process_data(
        input_folder=sys_config.data_root,
        preprocessing_folder=sys_config.preproc_folder,
        mode=exp_config.data_mode,
        size=exp_config.image_size,
        target_resolution=exp_config.target_resolution,
        force_overwrite=False
    )

    batch_size = 1

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
    #mask_pl, softmax_pl = model.predict(images_pl, exp_config.model_handle, exp_config.nlabels)
    training_time_placeholder = tf.placeholder(tf.bool, shape=[])
    logits = model.inference(images_pl, exp_config.model_handle, training=training_time_placeholder, nlabels=exp_config.nlabels)
    softmax_pl = tf.nn.softmax(logits)
    threshold = tf.constant(0.95, dtype=tf.float32)
    s = tf.multiply(tf.ones(shape=[1, 212, 212, 1]), threshold)
    softmax_pl = tf.concat([s, softmax_pl[..., 1:]], axis=-1)
    mask_pl = tf.arg_max(logits, dimension=-1)


    saver = tf.train.Saver()
    init = tf.global_variables_initializer()


    with tf.Session() as sess:

        sess.run(init)

        #checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'recursion_1_model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)

        for i in range(10, 20):
            ind = i #np.random.randint(data['images_test'].shape[0])

            x = data['images_test'][ind,...]
            y = data['masks_test'][ind,...]

            x = image_utils.reshape_2Dimage_to_tensor(x)
            y = image_utils.reshape_2Dimage_to_tensor(y)

            feed_dict = {
                images_pl: x,
                training_time_placeholder: False
            }

            mask_out, softmax_out = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)


            #postprocessing

            fig = plt.figure()
            ax1 = fig.add_subplot(251)
            ax1.imshow(np.squeeze(x), cmap='gray')
            ax2 = fig.add_subplot(252)
            ax2.imshow(np.squeeze(y))
            ax3 = fig.add_subplot(253)
            ax3.imshow(np.squeeze(mask_out))

            ax5 = fig.add_subplot(256)
            ax5.imshow(np.squeeze(softmax_out[..., 0]))
            ax6 = fig.add_subplot(257)
            ax6.imshow(np.squeeze(softmax_out[..., 1]))
            ax7 = fig.add_subplot(258)
            ax7.imshow(np.squeeze(softmax_out[..., 2]))
            ax8 = fig.add_subplot(259)
            ax8.imshow(np.squeeze(softmax_out[..., 3]))
            ax8 = fig.add_subplot(2, 5, 10)
            ax8.imshow(np.squeeze(softmax_out[..., 4]))

            plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a 2D network on slices from the test dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    args = parser.parse_args()

    base_path = sys_config.project_root

    model_path = os.path.join(base_path, args.EXP_PATH)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    init_iteration = main(exp_config=exp_config)