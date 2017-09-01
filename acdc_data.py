# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import glob
import numpy as np
import logging
import nibabel as nib
import gc
import h5py
from skimage import transform
from shutil import copyfile
from random_walker import segment, reduce_scribble_length
import utils
import image_utils
from random import sample

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Dictionary to translate a diagnosis into a number
# NOR  - Normal
# MINF - Previous myiocardial infarction (EF < 40%)
# DCM  - Dialated Cardiomypopathy
# HCM  - Hypertrophic cardiomyopathy
# RV   - Abnormal right ventricle (high volume or low EF)
diagnosis_dict = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

def crop_or_pad_slice_to_size(slice, nx, ny):

    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def prepare_data(input_folder, output_file, mode, size, target_resolution):

    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    assert (mode in ['2D', '3D']), 'Unknown mode: %s' % mode
    if mode == '2D' and not len(size) == 2:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '3D' and not len(size) == 3:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '2D' and not len(target_resolution) == 2:
        raise AssertionError('Inadequate number of target resolution parameters')
    if mode == '3D' and not len(target_resolution) == 3:
        raise AssertionError('Inadequate number of target resolution parameters')

    hdf5_file = h5py.File(output_file, "w")

    diag_list = {'test': [], 'train': []}
    height_list = {'test': [], 'train': []}
    weight_list = {'test': [], 'train': []}
    patient_id_list = {'test': [], 'train': []}
    cardiac_phase_list = {'test': [], 'train': []}

    file_list = {'test': [], 'train': []}
    num_slices = {'test': 0, 'train': 0}

    logging.info('Counting files and parsing meta data...')

    for folder in os.listdir(input_folder):

        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):

            train_test = 'test' if (int(folder[-3:]) % 5 == 0) else 'train'

            infos = {}
            for line in open(os.path.join(folder_path, 'Info.cfg')):
                label, value = line.split(':')
                infos[label] = value.rstrip('\n').lstrip(' ')

            patient_id = folder.lstrip('patient')

            for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                file_list[train_test].append(file)

                # diag_list[train_test].append(diagnosis_to_int(infos['Group']))
                diag_list[train_test].append(diagnosis_dict[infos['Group']])
                weight_list[train_test].append(infos['Weight'])
                height_list[train_test].append(infos['Height'])

                patient_id_list[train_test].append(patient_id)

                systole_frame = int(infos['ES'])
                diastole_frame = int(infos['ED'])

                file_base = file.split('.')[0]
                frame = int(file_base.split('frame')[-1])
                if frame == systole_frame:
                    cardiac_phase_list[train_test].append(1)  # 1 == systole
                elif frame == diastole_frame:
                    cardiac_phase_list[train_test].append(2)  # 2 == diastole
                else:
                    cardiac_phase_list[train_test].append(0)  # 0 means other phase

                nifty_img = nib.load(file)
                num_slices[train_test] += nifty_img.shape[2]

    # Write the small datasets
    for tt in ['test', 'train']:
        hdf5_file.create_dataset('diagnosis_%s' % tt, data=np.asarray(diag_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('weight_%s' % tt, data=np.asarray(weight_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('height_%s' % tt, data=np.asarray(height_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('patient_id_%s' % tt, data=np.asarray(patient_id_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('cardiac_phase_%s' % tt, data=np.asarray(cardiac_phase_list[tt], dtype=np.uint8))

    if mode == '3D':
        nx, ny, nz_max = size
        n_train = len(file_list['train'])
        n_test = len(file_list['test'])

    elif mode == '2D':
        nx, ny = size
        n_test = num_slices['test']
        n_train = num_slices['train']

    else:
        raise AssertionError('Wrong mode setting. This should never happen.')

    # Create datasets for images and masks
    data = {}

    for tt, num_points in zip(['test', 'train'], [n_test, n_train]):
        data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size), dtype=np.float32)
        data['masks_%s' % tt] = hdf5_file.create_dataset("masks_%s" % tt, [num_points] + list(size), dtype=np.uint8)

    mask_list = {'test': [], 'train': [] }
    img_list = {'test': [], 'train': [] }

    logging.info('Parsing image files')

    for train_test in ['test', 'train']:

        write_buffer = 0
        counter_from = 0

        for file in file_list[train_test]:

            logging.info('-----------------------------------------------------------')
            logging.info('Doing: %s' % file)

            file_base = file.split('.nii.gz')[0]
            file_mask = file_base + '_gt.nii.gz'

            img_dat = utils.load_nii(file)
            mask_dat = utils.load_nii(file_mask)

            img = img_dat[0].copy()
            mask = mask_dat[0].copy()

            img = image_utils.normalise_image(img)

            pixel_size = (img_dat[2].structarr['pixdim'][1],
                          img_dat[2].structarr['pixdim'][2],
                          img_dat[2].structarr['pixdim'][3])

            logging.info('Pixel size:')
            logging.info(pixel_size)

            ### PROCESSING LOOP FOR 3D DATA ################################
            if mode == '3D':

                scale_vector = [pixel_size[0] / target_resolution[0],
                                pixel_size[1] / target_resolution[1],
                                pixel_size[2]/ target_resolution[2]]

                img_scaled = transform.rescale(img,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               multichannel=False,
                                               mode='constant')
                mask_scaled = transform.rescale(mask,
                                                scale_vector,
                                                order=0,
                                                preserve_range=True,
                                                multichannel=False,
                                                mode='constant')

                slice_vol = np.zeros((nx, ny, nz_max), dtype=np.float32)
                mask_vol = np.zeros((nx, ny, nz_max), dtype=np.uint8)

                nz_curr = img_scaled.shape[2]
                stack_from = (nz_max - nz_curr) // 2

                if stack_from < 0:
                    raise AssertionError('nz_max is too small for the chosen through plane resolution. Consider changing'
                                         'the size or the target resolution in the through-plane.')

                for zz in range(nz_curr):

                    slice_rescaled = img_scaled[:,:,zz]
                    mask_rescaled = mask_scaled[:,:,zz]

                    slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                    mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, nx, ny)

                    slice_vol[:,:,stack_from] = slice_cropped
                    mask_vol[:,:,stack_from] = mask_cropped

                    stack_from += 1

                img_list[train_test].append(slice_vol)
                mask_list[train_test].append(mask_vol)

                write_buffer += 1

                if write_buffer >= MAX_WRITE_BUFFER:

                    counter_to = counter_from + write_buffer
                    _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
                    _release_tmp_memory(img_list, mask_list, train_test)

                    # reset stuff for next iteration
                    counter_from = counter_to
                    write_buffer = 0

            ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
            elif mode == '2D':

                scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]

                for zz in range(img.shape[2]):

                    slice_img = np.squeeze(img[:, :, zz])
                    slice_rescaled = transform.rescale(slice_img,
                                                       scale_vector,
                                                       order=1,
                                                       preserve_range=True,
                                                       multichannel=False,
                                                       mode = 'constant')

                    slice_mask = np.squeeze(mask[:, :, zz])
                    mask_rescaled = transform.rescale(slice_mask,
                                                      scale_vector,
                                                      order=0,
                                                      preserve_range=True,
                                                      multichannel=False,
                                                      mode='constant')

                    slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                    mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, nx, ny)

                    img_list[train_test].append(slice_cropped)
                    mask_list[train_test].append(mask_cropped)

                    write_buffer += 1

                    # Writing needs to happen inside the loop over the slices
                    if write_buffer >= MAX_WRITE_BUFFER:

                        counter_to = counter_from + write_buffer
                        _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
                        _release_tmp_memory(img_list, mask_list, train_test)

                        # reset stuff for next iteration
                        counter_from = counter_to
                        write_buffer = 0

        # after file loop: Write the remaining data

        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
        _release_tmp_memory(img_list, mask_list, train_test)


    # After test train loop:
    hdf5_file.close()


def _write_range_to_hdf5(hdf5_data, train_test, img_list, mask_list, counter_from, counter_to):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['masks_%s' % train_test][counter_from:counter_to, ...] = mask_arr


def _release_tmp_memory(img_list, mask_list, train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    mask_list[train_test].clear()
    gc.collect()

def most_recent_recursion(folder):
    '''
    Finds most largest recursion in an experiment based of data file
    N.B. May have made the data file for new recursion, but not finished processing it
    :param folder: experiment folder to check for recursions
    :return: recursion (int) - -1 if none is found
    '''
    max_recursion = 0
    recursion = -1
    for file in glob.glob(os.path.join(folder, 'recursion_*_data.hdf5')):
        file = file.split('/')[-1]
        _, recursion, _ = file.split('_')
        recursion = int(recursion)
        max_recursion = recursion if recursion > max_recursion else max_recursion

    return recursion

def load_and_maybe_process_scribbles(scribble_file, target_folder, percent_full_sup=0, scr_ratio=1):
    '''
    Loads in scribble data files, processing as necessary
    :param scribble_file: path of scribble file from
    :param target_folder:
    :param percent_full_sup:
    :return:
    '''
    #Check if data file already exists
    initialise = False
    current_recursion = most_recent_recursion(target_folder)
    if not os.path.exists(target_folder + '/base_data.hdf5'):
        initialise = True
    elif current_recursion == -1:
        initialise = True

    if initialise:
        logging.info('This experiment\'s scribble data has not yet been preprocessed')
        logging.info('Preprocessing now!')
        hdf_file = prepare_scribbles(scribble_file, target_folder, percent_full_sup, scr_ratio)
        current_recursion = 0
    else:
        hdf_file = h5py.File(recursion_filepath(current_recursion, folder_path=target_folder), 'r+')
        logging.info('Loaded in {}'.format(recursion_filepath(current_recursion, folder_path=target_folder)))

    base_data = h5py.File(os.path.join(target_folder, 'base_data.hdf5'), 'r')
    hdf_path = hdf_file.filename
    hdf_file.close()
    hdf_file = h5py.File(hdf_path, 'r')
    return base_data, hdf_file, current_recursion,

def recursion_filepath(recursion, folder_path = None, data_file = None):
    if folder_path is None:
        folder_path = os.path.dirname(data_file.filename)

    return os.path.join(folder_path, 'recursion_{}_data.hdf5'.format(recursion))

def load_different_recursion(data, step):
    recursion = utils.get_recursion_from_hdf5(data) + step
    return h5py.File(recursion_filepath(recursion, data_file=data))

def create_recursion_dataset(target_folder, recursion):
    base_data_file = h5py.File(os.path.join(target_folder, 'base_data.hdf5'), 'r')

    logging.info("Creating new dataset - recursion_{}_data.hdf5".format(recursion))

    #Create new file
    new_data_file = h5py.File(recursion_filepath(recursion,folder_path=target_folder))

    # prediction dataset: Contains the (thresholded, if option chosen) predictions after previous recursion
    new_data_file.create_dataset('predicted', dtype='uint8', shape=base_data_file['scribbles_train'].shape)
    new_data_file['predicted'].attrs.create('processed', False, dtype='bool')
    new_data_file['predicted'].attrs.create('processed_to', 0, dtype='uint16')

    # postprocessed dataset: Contains the postprocessed predictions after previous recursion
    new_data_file.create_dataset('postprocessed', dtype='uint8', shape=base_data_file['scribbles_train'].shape)
    new_data_file['postprocessed'].attrs.create('processed', False, dtype='bool')
    new_data_file['postprocessed'].attrs.create('processed_to', 0, dtype='uint16')

    # random_walked dataset:
    new_data_file.create_dataset('random_walked', dtype='uint8', shape=base_data_file['scribbles_train'].shape)
    new_data_file['random_walked'].attrs.create('processed', False, dtype='bool')
    new_data_file['random_walked'].attrs.create('processed_to', 0, dtype='uint16')

    base_data_file.close()
    return new_data_file

def prepare_scribbles(scribble_file, target_folder, percentage_full_sup, scr_ratio=1):
    '''
    Initialises data at the start of a run, copying scribble data and preparing
    data file of first recursion
    :param scribble_file: filename + path (from base) of scribble data file
    :param target_folder: folder (from base) of experiment directory
    :param percentage_full_sup: percentage (0 - 100) of images which should be taken from
                                fully annotated dataset
    :return:
    '''
    try:
        #Copy scribble hdf5
        print("Copying {} to {}".format(scribble_file, target_folder + '/base_data.hdf5'))

        copyfile(scribble_file, os.path.join(target_folder, 'base_data.hdf5'))
        base_data_file = h5py.File(os.path.join(target_folder, 'base_data.hdf5'), 'r+')

        ###FILE STRUCTURE:
        # For each recursion there will be a new data file (reduce loss if corruption occurs)
        # In each data file there are three datasets:
        # PREDICTED: The segmentations predicted by the previous recursion
        # POSTPROCESSED: The segmentations which have undergone postprocessing
        # RANDOM WALKED: The segmentations which have been random walked. These are th
        #                ground truths of the next recursion

        #Create file for first epoch
        pre_epoch_file = create_recursion_dataset(target_folder, 0)
        original_data = pre_epoch_file['postprocessed']
        source_scribbles = base_data_file['scribbles_train']

        if scr_ratio < 0 or scr_ratio >=1:
            original_data[...] = np.array(source_scribbles[...])
            pre_epoch_file['postprocessed'].attrs.modify('processed_to', len(original_data))
        else:
            for scr_idx in range(len(source_scribbles)):
                if scr_idx % 10 == 0:
                    #update
                    logging.info("Reducing length of scribbles {} to {}".format(scr_idx, scr_idx + 9))
                original_data[scr_idx, ...] = reduce_scribble_length(source_scribbles[scr_idx,...], ratio=scr_ratio)
                pre_epoch_file['postprocessed'].attrs.modify('processed_to', scr_idx)


        # Take some fully supervised data
        if percentage_full_sup > 0:
            #Get random indices for images which will be fully supervised
            num_slices = np.ceil(percentage_full_sup*len(original_data)/100).astype(np.int)
            logging.info("Taking {} slices from fully supervised annotations".format(num_slices))
            full_sup_indices = np.sort(np.unique(sample(range(len(original_data)), num_slices)))
            fs_masks = base_data_file['masks_train'][full_sup_indices, ...]
            fs_masks[fs_masks == 0] = np.amax(base_data_file['scribbles_train'])
            original_data[full_sup_indices, ...] = fs_masks

        #preprocess for first epoch
        logging.info('Processing random walk for first epoch')
        base_data_file.close()
        pre_epoch_file['postprocessed'].attrs.modify('processed',True)
        return pre_epoch_file
    except:
        logging.info('Scribble preparation failed. Files will be deleted.')
        os.remove(os.path.join(target_folder, 'base_data.hdf5')) if os.path.isfile(os.path.join(target_folder, 'base_data.hdf5')) else None
        os.remove(os.path.join(target_folder, 'recursion_0_data.hdf5')) if os.path.isfile(os.path.join(target_folder, 'recursion_0_data.hdf5')) else None
        return 0

def random_walk_epoch(hdf_file, beta, threshold, random_walk=True):
    if not 'postprocessed' in hdf_file:
        logging.warning('Attempted to random walk for data file which '
                        'does not have postprocessed predictions from '
                        'previous epoch')
    else:
        #reopen data file in read mode
        data_fpath = hdf_file.filename
        hdf_file.close()
        hdf_file = h5py.File(data_fpath, 'r+')

        #get images from base data file
        base_data_file = h5py.File(os.path.join(os.path.dirname(hdf_file.filename), 'base_data.hdf5'), 'r')
        images = np.array(base_data_file['images_train'])
        base_data_file.close()

        #get scribble data as output of previous epoch
        seeds = hdf_file['postprocessed']
        random_walked = hdf_file['random_walked']

        #get checkpoint metadata
        processed = random_walked.attrs.get('processed')
        processed_to = random_walked.attrs.get('processed_to')
        recursion = utils.get_recursion_from_hdf5(hdf_file)
        if not processed:
            #process in batches of 20
            #doesn't really make a time difference
            logging.info("Random walking for recursion {}".format(recursion))
            batch_size = 20
            for scr_idx in range(processed_to, len(seeds), batch_size):
                if random_walk:
                    logging.info('Random walking range {} to {} of recursion {}'.format(scr_idx, scr_idx+ batch_size - 1, recursion))
                    random_walked[scr_idx:scr_idx+batch_size, ...] = segment(images[scr_idx:scr_idx+batch_size, ...],
                                                                             seeds[scr_idx:scr_idx+batch_size, ...],
                                                                             beta=beta,
                                                                             threshold=threshold)
                else:
                    random_walked[scr_idx:scr_idx+batch_size, ...] = seeds[scr_idx:scr_idx+batch_size, ...]

                random_walked.attrs.modify('processed_to', scr_idx + batch_size)

            random_walked.attrs.modify('processed', True)

        #reopen in read mode
        hdf_file.close()
        hdf_file = h5py.File(data_fpath, 'r')
    return hdf_file

def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                mode,
                                size,
                                target_resolution,
                                force_overwrite=False):

    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data
    
    :param input_folder: Folder where the raw ACDC challenge data is located
    :param preprocessing_folder: Folder where the preprocessed data should be written to
    :param mode: Can either be '2D' or '3D'. 2D saves the data slice-by-slice, 3D saves entire volumes
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]
    :param weak_supervision: Set this to True if data is scribbles
    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in np.round(target_resolution,2)])

    data_file_name = 'data_%s_size_%s_res_%s.hdf5' % (mode, size_str, res_str)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, mode, size, target_resolution)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')



if __name__ == '__main__':

    input_folder = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617'
    preprocessing_folder = 'preproc_data'

    # d=load_and_maybe_process_data(input_folder, preprocessing_folder, '3D', (116,116,28), (2.5,2.5,5))
    d=load_and_maybe_process_data(input_folder, preprocessing_folder, '2D', (212,212), (1.36719, 1.36719))

