# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import nibabel as nib
import numpy as np
import os
import glob

def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

def get_latest_model_checkpoint_path(folder, name):
    '''
    Returns the checkpoint with the highest iteration number with a given name
    :param folder: Folder where the checkpoints are saved
    :param name: Name under which you saved the model
    :return: The path to the checkpoint with the latest iteration
    '''

    iteration_nums = []
    file_bases = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):
        print(file)
        file = file.split('/')[-1]
        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])
        file_bases.append(file_base)
        iteration_nums.append(it_num)
    latest_iteration = np.max(iteration_nums)
    ind_max = np.argmax(iteration_nums)
    #return os.path.join(folder, name + '-' + str(latest_iteration))
    return os.path.join(folder, file_bases[ind_max] + '.' + name.split('.')[-1] + '-' + str(latest_iteration))

def get_recursion_from_hdf5(data):
    '''
    Helper function to extract recursion number from a datafile
    :param data: weak supervision data file with filename format 'recursion_n_data.hdf5'
    :return: recursion number
    '''
    recursion = os.path.basename(data.filename)
    _, recursion, _ = recursion.split('_')
    return int(recursion)
