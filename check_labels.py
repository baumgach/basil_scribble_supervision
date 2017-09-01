__author__ = 'bmustafa'


import h5py
import config.system as sys_config
import os
import numpy as np
import matplotlib.pyplot as plt
from random_walker import segment as rw
from time import time

f = h5py.File(os.path.join(sys_config.project_root, 'preproc_data/data.hdf5'),'r')

scribbles = f['scribbles_train']
images = f['images_train']
masks = f['masks_train']

num_to_look_for = 2
fig = plt.figure()
for i in range(len(scribbles)):
    show = False
    x = np.sort(np.unique(scribbles[i, ...]))
    if len(x) <= num_to_look_for:
        print("Slice {} has {} labels - {}".format(i, len(x), x))
        show = True

    if x[0] != 0:
        print("Slice {} has no 0 label".format(i))
        show = True

    show = False
    if show:
        ax = fig.add_subplot(221)
        ax.imshow(np.squeeze(images[i, ...]), cmap='gray')

        ax = fig.add_subplot(222)
        ax.imshow(np.squeeze(scribbles[i, ...]))

        ax = fig.add_subplot(223)
        ax.imshow(np.squeeze(masks[i, ...]))

        #ax = fig.add_subplot(224)
        #ax.imshow(np.squeeze(rw(images[i, ...], scribbles[i, ...], beta=1000, threshold=0.99)))
        t = time()

