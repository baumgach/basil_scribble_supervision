
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
import h5py
import config.system as sys_config
import utils
from image_utils import print_coloured, print_grayscale
import image_utils
import matplotlib.pyplot as plt
import model
from random_walker import segment
import acdc_data
from medpy.metric.binary import dc

OUTPUT_FOLDER = sys_config.project_root + 'poster/images/'
fig = plt.figure()
def dice(result, reference):
    d = 0
    res = np.squeeze(result)
    ref = np.squeeze(reference)
    for layer in range(1, 3):
        d+=dc(res==layer, ref==layer)

    return d/3



def main(ws_exp_config, slices, test):
    # Load data
    exp_dir = sys_config.project_root + 'acdc_logdir/' + ws_exp_config.experiment_name + '/'
    base_data = h5py.File(os.path.join(exp_dir, 'base_data.hdf5'), 'r')

    # Get number of recursions
    num_recursions = acdc_data.most_recent_recursion(sys_config.project_root + 'acdc_logdir/' + ws_exp_config.experiment_name)
    print(num_recursions)

    num_recursions+=1
    # Get images
    batch_size = len(slices)

    if test:
        slices = slices[slices < len(base_data['images_test'])]
        images = base_data['images_test'][slices, ...]
        gt = base_data['masks_test'][slices, ...]
        prefix='test'
    else:
        slices = slices[slices < len(base_data['images_train'])]
        images = base_data['images_train'][slices, ...]
        gt = base_data['masks_train'][slices, ...]
        scr = base_data['scribbles_train'][slices, ...]
        prefix='train'

    image_tensor_shape = [batch_size] + list(ws_exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
    feed_dict = {
        images_pl: np.expand_dims(images, -1),
    }

    #Get weak supervision predictions
    mask_pl, softmax_pl = model.predict(images_pl, ws_exp_config.model_handle, ws_exp_config.nlabels)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    predictions = np.zeros([batch_size] + list(ws_exp_config.image_size) + [num_recursions])
    predictions_klc = np.zeros_like(predictions)
    predictions_rw = np.zeros_like(predictions)
    with tf.Session() as sess:
        sess.run(init)
        for recursion in range(num_recursions):
            try:
                try:
                    checkpoint_path = utils.get_latest_model_checkpoint_path(ws_model_path,
                                                                             'recursion_{}_model_best_xent.ckpt'.format(recursion))
                except:
                    try:
                        checkpoint_path = utils.get_latest_model_checkpoint_path(ws_model_path,
                                                                                 'recursion_{}_model_best_dice.ckpt'.format(recursion))
                    except:
                        checkpoint_path = utils.get_latest_model_checkpoint_path(ws_model_path,
                                                                                 'recursion_{}_model.ckpt'.format(recursion))
                saver.restore(sess, checkpoint_path)
                mask_out, _ = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)
                predictions[..., recursion] = mask_out
                for i in range(batch_size):
                    predictions_klc[i, :, :, recursion] = image_utils.keep_largest_connected_components(mask_out[i, ...])

                predictions_rw[..., recursion] = segment(images, np.squeeze(predictions_klc[..., recursion]), beta=ws_exp_config.rw_beta, threshold=0)

                print("Classified for recursion {}".format(recursion))
            except Exception:
                predictions[..., recursion] = -1*np.zeros_like(predictions[..., recursion])
                print("Could not find checkpoint for recursion {} - skipping".format(recursion))


    for i in range(batch_size):
        pref = '{}{}'.format(prefix, slices[i])

        print_grayscale(images[i, ...], filepath=OUTPUT_FOLDER, filename='{}_image'.format(pref))
        print_coloured(gt[i, ...],  filepath=OUTPUT_FOLDER, filename='{}_gt'.format(pref))
        for recursion in range(num_recursions):
            if np.max(predictions[i, :, :, recursion]) >= -0.5:
                print_coloured(predictions[i, :, :, recursion],  filepath=OUTPUT_FOLDER, filename="{}_ws_pred_r{}".format(pref, recursion))
                print_coloured(predictions_klc[i, :, :, recursion],  filepath=OUTPUT_FOLDER, filename="{}_ws_pred_klc_r{}".format(pref, recursion))
                print_coloured(predictions_rw[i, :, :, recursion],  filepath=OUTPUT_FOLDER, filename="{}_ws_pred_klc_rw_r{}".format(pref,recursion))
                print("Dice coefficient for slice {} is {}".format(slices[i],
                                                                   dice(predictions_rw[i, :, :, recursion], gt[i, ...])))
        if not test:
            print_coloured(scr[i, ...],  filepath=OUTPUT_FOLDER, filename='{}_scribble'.format(pref))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a 2D network on slices from the test dataset")
    parser.add_argument("WS_EXP_PATH", type=str, help="Path to weakly supervised experiment folder "
                                                   "(assuming you are in the working directory)")

    parser.add_argument("TEST_DATA", type=int, help="If 1, will use test data. Else will use training data")

    parser.add_argument("SLICE_NUMBERS", type=int, help="Indices of desired images", nargs="+")

    args = parser.parse_args()
    base_path = sys_config.project_root

    ws_model_path = os.path.join(base_path, args.WS_EXP_PATH)
    ws_config_file = glob.glob(ws_model_path + '/*py')[0]
    ws_config_module = ws_config_file.split('/')[-1].rstrip('.py')

    ws_exp_config = SourceFileLoader(ws_config_module, os.path.join(ws_config_file)).load_module()

    test_data = (args.TEST_DATA == 1)

    main(ws_exp_config=ws_exp_config,
         slices=np.sort(np.unique(args.SLICE_NUMBERS)),
         test=test_data)