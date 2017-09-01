
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
import h5py
import config.system as sys_config
import utils
import image_utils
import model
from acdc_data import most_recent_recursion
from random_walker import segment

def main(exp_config, batch_size = 3):

    # Load data
    data = h5py.File(sys_config.project_root + exp_config.scribble_data, 'r')



    slices = np.random.randint(low=0, high=data['images_test'].shape[0], size=batch_size)
    slices = np.sort(np.unique(slices))
    slices = [80, 275, 370]
    batch_size = len(slices)
    images = data['images_test'][slices, ...]
    masks = data['masks_test'][slices, ...]
    #masks[masks == 0] = 4

    num_recursions = most_recent_recursion(model_path)

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
    mask_pl, softmax_pl = model.predict(images_pl, exp_config.model_handle, exp_config.nlabels)
    #mask_fs_pl, softmax_fs_pl = model.predict(images_pl,  unet2D_bn_modified, 4)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    predictions = np.zeros([batch_size] + list(exp_config.image_size) + [num_recursions + 1])

    feed_dict = {
        images_pl: np.expand_dims(images, -1),
    }
    path = '/scratch_net/'
    with tf.Session() as sess:
        sess.run(init)
        pred_size = 0
        for recursion in range(num_recursions + 1):
            try:
                try:
                    checkpoint_path = utils.get_latest_model_checkpoint_path(model_path,
                                                                             'recursion_{}_model_best_dice.ckpt'.format(recursion))
                except:
                    checkpoint_path = utils.get_latest_model_checkpoint_path(model_path,
                                                                             'recursion_{}_model.ckpt'.format(recursion))
                saver.restore(sess, checkpoint_path)
                mask_out, _ = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)
                for mask in range(batch_size):
                    predictions[mask, ..., pred_size] = image_utils.keep_largest_connected_components(np.squeeze(mask_out[mask, ...]))
                print("Classified for recursion {}".format(recursion))
                pred_size+=1
            except Exception:
                print("Could not find checkpoint for recursion {} - skipping".format(recursion))

    num_recursions = pred_size
    fig = plt.figure()
    num_cols = num_recursions + 3
    #RW:
    path = base_path + "/poster/"
    for recursion in range(num_recursions):
        predictions[..., recursion] = segment(images, np.squeeze(predictions[..., recursion]), beta=exp_config.rw_beta, threshold=0)

    for r in range(batch_size):
        #Add the image
        # ax = fig.add_subplot(batch_size, num_cols, 1 + r*num_cols)
        # ax.axis('off')
        # ax.imshow(np.squeeze(images[r, ...]), cmap='gray')
        image_utils.print_grayscale(np.squeeze(images[r,...]),path, '{}_{}_image'.format(exp_config.experiment_name, slices[r]))
        #Add the mask
        # ax = fig.add_subplot(batch_size, num_cols, 2 + r*num_cols)
        # ax.axis('off')
        # ax.imshow(np.squeeze(masks[r, ...]), vmin=0, vmax=4, cmap='jet')
        image_utils.print_coloured(np.squeeze(masks[r, ...]),path, '{}_{}_gt'.format(exp_config.experiment_name, slices[r]))

        #predictions[r, ...] = segment(images, np.squeeze(predictions[r, ...]), beta=exp_config.rw_beta, threshold=0)
        for recursion in range(num_recursions):
            #Add each prediction
            image_utils.print_coloured(np.squeeze(predictions[r, ..., recursion]),path, '{}_{}_pred_r{}'.format(exp_config.experiment_name, slices[r], recursion))
            #ax = fig.add_subplot(batch_size, num_cols, 3 + recursion + r*num_cols)
            #ax.axis('off')
            #ax.imshow(np.squeeze(predictions[r, ..., recursion]), vmin=0, vmax=4, cmap='jet')

    while True:
        plt.axis('off')
        plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a 2D network on slices from the test dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    parser.add_argument("BATCH_SIZE", type=int, help="Number of images to display")
    args = parser.parse_args()

    base_path = sys_config.project_root

    model_path = os.path.join(base_path, args.EXP_PATH)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    init_iteration = main(exp_config=exp_config, batch_size=args.BATCH_SIZE)