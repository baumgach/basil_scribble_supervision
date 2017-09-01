import glob
import argparse
import logging
import tensorflow as tf
import numpy as np
import os
from importlib.machinery import SourceFileLoader
import model
import h5py
import config.system as sys_config
from random_walker import segment as rw
import acdc_data
from random import sample
import utils
from matplotlib import pyplot as plt


def dice2D(mask_1, mask_2,  bg_label=None):
    assert mask_1.shape == mask_2.shape, "Shapes of mask 1 ({}) does not match shape of mask 2 ({})"

    #get labels

    max_label = np.max(np.unique([mask_1,mask_2])) if bg_label is None else bg_label
    labels = list(range(0, max_label + 1))
    if not bg_label is None:
        labels.remove(bg_label)
    labels.remove(0)
    labels = np.array(labels)

    #calculate
    dices = np.zeros(labels.shape[0], dtype=np.double)
    d_idx = 0
    for label in labels:
        #calculate union
        union = np.sum(np.any((mask_1 == label, mask_2 == label), axis=0))
        intersect = np.sum(np.all((mask_1 == label, mask_2 == label), axis=0))
        if intersect == 0:
            dices[d_idx] = 0
        elif union == 0:
            dices[d_idx] = np.nan
        else:
            dices[d_idx] = intersect/union
        d_idx+=1

    return dices, np.nanmean(dices)

def calculate_dices(predictions, segmentations, nlabels, path=None, filename=None):
    #initialise variables
    mean_dices = np.zeros([len(predictions)])
    dice_scores = np.zeros([len(predictions), nlabels - 2])
    for i, prediction in enumerate(predictions):
        dice_scores[i, ...], mean_dices[i] = dice2D(prediction, segmentations[i, ...], nlabels - 1)

    sorted_dices = np.argsort(mean_dices)
    indices = [sorted_dices[0],
               sorted_dices[int(len(mean_dices)/2)],
               sorted_dices[-1]]
    if not path is None:
        headerstr = ""
        for i in range(1, nlabels):
            headerstr += "Dice {:02},".format(i)
        headerstr +="Mean Dice"
        file = open(path + filename + ".csv", 'w')
        file.close()

        np.savetxt(fname=path + filename + ".csv",
                   X=np.concatenate((dice_scores, np.expand_dims(mean_dices, axis=1)), axis=1),
                   delimiter=",",
                   header=headerstr)
    stdevs = np.std(np.concatenate((dice_scores, np.expand_dims(mean_dices, axis=1)), axis=1), axis=0)
    return indices, np.nanmean(dice_scores, axis=0), np.mean(mean_dices), stdevs

def initialise_dice_attrs(data, path, nlabels):
    data[path].attrs.create('mean_dice',
                            dtype=np.float,
                            data=-1)
    placeholder = np.zeros(nlabels - 2, dtype=np.float)
    data[path].attrs.create('dices',
                            dtype=np.float,
                            shape=(nlabels - 2,),
                            data=placeholder)
    placeholder = np.zeros(nlabels - 1, dtype=np.float)
    data[path].attrs.create('stdevs',
                            dtype=np.float,
                            shape=(nlabels - 1,),
                            data=placeholder)
def dice_str(data):
    dices = data.attrs.get('dices')
    mean = data.attrs.get('mean_dice')
    stdevs = data.attrs.get('stdevs')

    t_str = ""
    for ind, (dice, stdev) in enumerate(zip(dices, stdevs[0:-1])):
        t_str += "{0:.3f} ({1:.3f})".center(20).format(dice, stdev)

    t_str+="{:.3f} ({:.3f})".center(24).format(mean, stdevs[-1])
    return t_str

def main(csv_path=None, batch_size=None):
    #get number of recursions
    num_recursions = acdc_data.most_recent_recursion(model_path)

    #get data
    base_data = h5py.File(os.path.join(model_path, 'base_data.hdf5'), 'r')

    masks_gt = np.array(base_data['masks_train'])
    images = np.array(base_data['images_train'])
    base_data.close()
    r_data = h5py.File(os.path.join(model_path, 'recursion_0_data.hdf5'), 'r')
    scribbles = np.array(r_data['predicted'])
    r_data.close()

    with h5py.File(model_path + "/recursion_evaluation/assessment.hdf5") as output_data:
        #FULLY RANDOM WALKED
        if not 'random_walk_mask' in output_data:
            output_data.create_dataset(name='random_walk_mask',
                                       dtype=np.uint8,
                                       shape=images.shape)
            output_data['random_walk_mask'].attrs.create('processed_to',
                                                         dtype=np.uint16,
                                                         data=0)
            initialise_dice_attrs(output_data, 'random_walk_mask', exp_config.nlabels)

        #process random walk
        processed_to = output_data['random_walk_mask'].attrs.get('processed_to') + 1
        for scr_idx in range(processed_to, len(images), exp_config.batch_size):
            ind = list(range(scr_idx, min(scr_idx + exp_config.batch_size, len(images))))
            logging.info('Preparing random walker segmentation for slices {} to {}'.format(ind[0], ind[-1]))
            output_data['random_walk_mask'][ind, ...] = scribbles[ind, ...]
            # output_data['random_walk_mask'][ind, ...] = rw(images[ind, ...],
            #                                                scribbles[ind, ...],
            #                                                beta=exp_config.rw_beta,
            #                                                threshold=0)
            # output_data['random_walk_mask'].attrs.modify('processed_to', ind[-1] + 1)

        #calculate dice
        _, dices, mean_dice, stdevs = calculate_dices(np.array(output_data['random_walk_mask']),
                                                      masks_gt,
                                                      exp_config.nlabels,
                                                      path=csv_path,
                                                      filename="fully_random_walked")
        output_data['random_walk_mask'].attrs.modify('mean_dice', mean_dice)
        output_data['random_walk_mask'].attrs.modify('dices', dices)
        output_data['random_walk_mask'].attrs.modify('stdevs', stdevs)

        #iterate through recursions
        #recursion 0 slightly different as it has no 'output from previous epoch' set
        r_data = h5py.File(os.path.join(model_path, 'recursion_0_data.hdf5'),'r')
        dset_names = np.array([['recursion_{0}/processed_{0}',
                                'recursion_{0}/random_walked_{0}',
                                'recursion_{0}/prediction_{0}'],
                               ['postprocessed',
                                'random_walked',
                                'predicted'],
                               ['processed r{0} seg',
                                'random walked r{0} seg',
                                'r{0} seg']
                               ])

        for recursion in range(0, num_recursions + 1):
            for i in range(3):
                dset_name = dset_names[0, i].format(recursion)
                #once it gets to the prediction it must open the next dataset file
                if i == 2:
                    r_data.close()
                    if recursion == num_recursions:
                        break
                    r_data = h5py.File(os.path.join(model_path, 'recursion_{}_data.hdf5'.format(recursion + 1)), 'r')

                if not dset_name in output_data:
                    logging.info("Creating dataset {}".format(dset_name))
                    logging.info("Getting data from {}[{}]".format(os.path.basename(r_data.filename).split('.')[0], dset_names[1,i]))
                    output_data.create_dataset(dset_name,
                                               dtype=np.uint8,
                                               shape=images.shape,
                                               data=r_data[dset_names[1, i]])
                    initialise_dice_attrs(output_data, dset_name, exp_config.nlabels)
                else:
                    logging.info("Getting data from {}[{}]".format(os.path.basename(r_data.filename).split('.')[0], dset_names[1,i]))
                    output_data[dset_name][:] = np.array(r_data[dset_names[1, i]])
                #calculate dices
                indices, dices, mean_dice, stdevs = calculate_dices(np.array(output_data[dset_name]),
                                                                    masks_gt,
                                                                    exp_config.nlabels,
                                                                    path=csv_path,
                                                                    filename=dset_names[2,i].format(recursion))

                output_data[dset_name].attrs.modify('mean_dice', mean_dice)
                output_data[dset_name].attrs.modify('dices', dices)
                output_data[dset_name].attrs.modify('stdevs', stdevs)

        #for last recursion, need to use prediction from network
        final_dset = 'recursion_{0}/prediction_{0}'.format(recursion)
        if not final_dset in output_data:
            output_data.create_dataset(final_dset,
                                       dtype=np.uint8,
                                       shape=images.shape,
                                       data=masks_gt)
            output_data[final_dset].attrs.create(name='processed_to',
                                                 data=np.array((0,)),
                                                 shape=(1,),
                                                 dtype=np.uint16)
            output_data[final_dset].attrs.create(name='processed',
                                                 data=np.array(False),
                                                 shape=(),
                                                 dtype=np.bool)
            initialise_dice_attrs(output_data, final_dset, exp_config.nlabels)


        try:
            checkpoint_path = utils.get_latest_model_checkpoint_path(model_path,
                                                                     'recursion_{}_model*.ckpt'.format(recursion))
            skip_final_recursion = False
        except:
            skip_final_recursion = True

        if not skip_final_recursion:
            image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [1]
            images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
            mask_pl, softmax_pl = model.predict(images_pl, exp_config.model_handle, exp_config.nlabels)
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            training_time_placeholder = tf.placeholder(tf.bool, shape=[])

            with tf.Session() as sess:
                sess.run(init)
                try:
                    saver.restore(sess, checkpoint_path)
                    epoch = int(checkpoint_path.split('/')[-1].split('-')[-1]) + 1
                    epoch = int(epoch / (len(images)/exp_config.batch_size))
                    epoch = epoch % exp_config.epochs_per_recursion
                    skip_final_recursion = False
                except:
                    logging.info("Failed to load checkpoint for recursion {}".format(recursion))
                    skip_final_recursion = True

                if not skip_final_recursion:
                    scr_max = len(images)
                    processed_to = output_data[final_dset].attrs.get('processed_to')
                    print(processed_to)
                    processed_to = 0 if processed_to is None else processed_to
                    processed_to = np.squeeze(processed_to)

                    for scr_idx in range(processed_to, len(images), exp_config.batch_size):
                        if scr_idx+exp_config.batch_size > scr_max:
                            # At the end of the dataset
                            ind = list(range(scr_max - exp_config.batch_size, scr_max))
                        else:
                            ind = list(range(scr_idx, scr_idx + exp_config.batch_size))

                        logging.info("Segmenting images using weights from final "
                                     "recursion ({}) for slices {} to {}".format(recursion, ind[0], ind[-1]))
                        feed_dict = {
                            images_pl: np.expand_dims(images[ind, ...], -1),
                            training_time_placeholder: False
                        }
                        output_data[final_dset][ind, ...] = scribbles[ind, ...]
                        # output_data[final_dset][ind, ...], _ = sess.run([mask_pl, softmax_pl],
                        #                                                 feed_dict=feed_dict)
                        output_data[final_dset].attrs.modify('processed_to', ind[-1] + 1)

                    #Calculate dices
                    _, dices, mean_dice, stdevs = calculate_dices(np.array(output_data[final_dset]),
                                                          masks_gt,
                                                          exp_config.nlabels,
                                                          path=csv_path,
                                                          filename=dset_names[2, 2].format(recursion))
                    output_data[final_dset].attrs.modify('mean_dice', mean_dice)
                    output_data[final_dset].attrs.modify('dices', dices)
                    output_data[final_dset].attrs.modify('stdevs', stdevs)
                    output_data[final_dset].attrs.modify('processed', True)

        #print summaries:
        print("DICES:")
        l_str = " "* 40
        for i in range(1, exp_config.nlabels -1):
            d_str = "Dice 0{} (stdev)".format(i)
            print(d_str)
            print(d_str.center(20))
            l_str += d_str.center(20)
        print(l_str + "       Mean Dice        ")

        print("      Random walker segmentations:".ljust(45) + dice_str(output_data['random_walk_mask']))
        for recursion in range(0, num_recursions + 1):
            print("   RECURSION {}".format(recursion))
            print("      Postprocessed input of recursion {}".format(recursion).ljust(45) +
                         dice_str(output_data[dset_names[0,0].format(recursion)]))

            print("      Random walked input of recursion {}".format(recursion).ljust(45) +
                         dice_str(output_data[dset_names[0,1].format(recursion)]))
            if recursion != num_recursions:
                print("      Output of recursion {}".format(recursion).ljust(45) +
                             dice_str(output_data[dset_names[0,2].format(recursion)]))

        #Handle last recursion seperately
        if not skip_final_recursion:
            print("      Output of recursion {}".format(recursion).ljust(45) +
                         dice_str(output_data[dset_names[0,2].format(recursion)]))
            print("         Weights for predicting final masks were taken from epoch {} of {}".format(
                epoch, exp_config.epochs_per_recursion))

        #get graphs       Mean Dice
        if not batch_size is None:
            indices = np.sort(np.unique(sample(range(len(images)), batch_size)))
            logging.info("Showing segmentation progression for slices randomly picked slices {}".format(indices))
            descriptions = []
            for index in indices:
                descriptions.append("Slice {:04}".format(index))
        else:
            logging.info("Showing segmentation progression for slices {0} [Best], {1} [Median] and {2} [Worst]".format(
                indices[0], indices[1], indices[2]
            ))
            descriptions = ['best slice ({:04})'.format(indices[0]),
                            'median slice ({:04})'.format(indices[1]),
                            'worst slice ({:04})'.format(indices[2])]
            batch_size = 3

        for fig_idx in range(batch_size):
            graphs = np.zeros((6 + num_recursions*3, exp_config.image_size[0], exp_config.image_size[1]))
            slice_idx = indices[fig_idx]
            #First get image, ground truth and random walked prediction
            graphs[1, ...] = masks_gt[slice_idx, ...]
            graphs[2, ...] = output_data['random_walk_mask'][slice_idx, ...]

            for recursion in range(num_recursions + 1):
                for i in range(3):
                    if i == 2 and recursion == num_recursions and skip_final_recursion:
                        break
                    dset_name = dset_names[0, i].format(recursion)
                    graphs[recursion*3 + i + 3, ...] = output_data[dset_name][slice_idx, ...]

            fig = plt.figure(fig_idx)
            fig.suptitle('Segmentation progress for {}'.format(descriptions[fig_idx]))
            #handle image seperately

            ax = fig.add_subplot(num_recursions + 2, 3, 1)
            ax.axis('off')
            ax.set_title('image')
            ax.imshow(np.squeeze(images[slice_idx, ...]), cmap='gray', vmin=0, vmax=exp_config.nlabels - 1)

            for graph_idx in range(1,len(graphs)):
                ax = fig.add_subplot(num_recursions + 2, 3, graph_idx + 1)
                ax.axis('off')

                # This should be cleaned up
                if graph_idx == 1:
                    ax.set_title('ground truth')
                elif graph_idx == 2:
                    ax.set_title('random walker segmentation')
                elif graph_idx == 3:
                    ax.set_title('weak supervision')
                elif graph_idx == 4:
                    ax.set_title('ws random walked')
                else:
                    ax.set_title(dset_names[2, graph_idx % 3].format(int((graph_idx + 1)/3) - 2))
                ax.imshow(np.squeeze(graphs[graph_idx, ...]),
                          cmap='jet',
                          vmin=0,
                          vmax=exp_config.nlabels - 1)


        plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a 2D network on slices from the test dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")

    parser.add_argument("SAVE_DICES",
                        help="OPTIONAL: Set to 1 to save a csv file of dices",
                        nargs='?',
                        default=argparse.SUPPRESS)
    parser.add_argument("--save", dest="SAVE_DICES", default=None)

    parser.add_argument("BATCH_SIZE",
                        help="OPTIONAL: Specify number of images to show. One figure will show for each image. \n"
                             "If not specified, will show best/worst/median",
                        nargs='?',
                        default=argparse.SUPPRESS)
    parser.add_argument("--num_images", dest="BATCH_SIZE", default=None)

    args = parser.parse_args()
    base_path = sys_config.project_root

    model_path = os.path.join(base_path, args.EXP_PATH)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')


    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    ##SAVE PATH
    save_path = model_path + "/recursion_evaluation/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    batch_size = None if args.BATCH_SIZE is None else int(args.BATCH_SIZE)
    if args.SAVE_DICES:
        main(csv_path=save_path, batch_size=batch_size)
    else:
        main(batch_size=batch_size)

