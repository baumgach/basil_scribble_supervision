
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
import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse
from image_utils import print_coloured
import config.system as sys_config
import utils
from acdc_data import load_and_maybe_process_data
import matplotlib.pyplot as plt

import model
from experiments import unet2D_bn_modified_xent as fs_exp_config
OUTPUT_FOLDER = sys_config.project_root + 'poster/images'
fig = plt.figure()

def main(fs_exp_config, slices, test):
    # Load data
    data = load_and_maybe_process_data(
        input_folder=sys_config.data_root,
        preprocessing_folder=sys_config.preproc_folder,
        mode=fs_exp_config.data_mode,
        size=fs_exp_config.image_size,
        target_resolution=fs_exp_config.target_resolution,
        force_overwrite=False
    )
    # Get images
    batch_size = len(slices)
    if test:
        slices = slices[slices < len(data['images_test'])]
        images = data['images_test'][slices, ...]
        prefix = 'test'
    else:
        slices = slices[slices < len(data['images_train'])]
        images = data['images_train'][slices, ...]
        prefix = 'train'

    image_tensor_shape = [batch_size] + list(fs_exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
    feed_dict = {
        images_pl: np.expand_dims(images, -1),
    }

    #Get full supervision prediction
    mask_pl, softmax_pl = model.predict(images_pl, fs_exp_config.model_handle, fs_exp_config.nlabels)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        checkpoint_path = utils.get_latest_model_checkpoint_path(fs_model_path,
                                                                 'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)
        fs_predictions, _ = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)

    for i in range(batch_size):
        print_coloured(fs_predictions[i, ...],  filepath=OUTPUT_FOLDER, filename='{}{}_fs_pred'.format(prefix, slices[i]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a 2D network on slices from the test dataset")
    parser.add_argument("FS_EXP_PATH", type=str, help="Path to fully supervised experiment folder "
                                                      "(assuming you are in the working directory)")

    parser.add_argument("TEST_DATA", type=int, help="If 1, will use test data. Else will use training data")

    parser.add_argument("SLICE_NUMBERS", type=int, help="Indices of desired images", nargs="+")
    args = parser.parse_args()

    test_data = (args.TEST_DATA == 1)

    base_path = sys_config.project_root
    fs_model_path = os.path.join(base_path, args.FS_EXP_PATH)
    fs_config_file = glob.glob(fs_model_path + '/*py')[0]
    fs_config_module = fs_config_file.split('/')[-1].rstrip('.py')
    fs_exp_config = SourceFileLoader(fs_config_module, os.path.join(fs_config_file)).load_module()

    main(fs_exp_config=fs_exp_config,
         slices=np.sort(np.unique(args.SLICE_NUMBERS)),
         test=test_data)