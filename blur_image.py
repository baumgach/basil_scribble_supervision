__author__ = 'bmustafa'
from scipy.ndimage.filters import gaussian_filter
import h5py
from matplotlib import pyplot as plt
import numpy as np

#PARAMS
SIGMA = 1.5
THRESHOLD = 0.6
f = h5py.File('/scratch_net/biwidl102/bmustafa/acdc_segmenter_internal/acdc_logdir/unet2D_ws_lr0/recursion_1_data.hdf5', 'r')

mask = f['postprocessed'][475, ...]

labels = np.unique(mask)
labels = labels[labels != 0]
labels = labels[labels != np.max(labels)]

fig = plt.figure()
new_mask = np.zeros_like(mask)
#new_mask = np.zeros(mask.shape)
for label in labels:
    struct = (mask == label).astype(np.float)
    blurred_struct = gaussian_filter(struct, sigma=SIGMA)
    # ax = fig.add_subplot(161 + label)
    blurred_struct[blurred_struct >= THRESHOLD] = 1
    blurred_struct[blurred_struct < THRESHOLD] = 0
    new_mask[blurred_struct != 0] = label

    # ax.imshow(blurred_struct + label - 1, vmin=0, vmax=4, cmap='jet')


ax = fig.add_subplot(121)
ax.imshow(mask, vmin=0, vmax=4, cmap='jet')
ax = fig.add_subplot(122)
ax.imshow(new_mask, vmin=0, vmax=4, cmap='jet')
# while True:
plt.show()