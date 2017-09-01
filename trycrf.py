__author__ = 'bmustafa'

import h5py
from matplotlib import pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from experiments import unet2D_ws_lr0 as exp_config
base_file = h5py.File('/scratch_net/biwidl102/bmustafa/acdc_segmenter_internal/acdc_logdir/unet2D_ws_lr0/base_data.hdf5','r')
data_file = h5py.File('/scratch_net/biwidl102/bmustafa/acdc_segmenter_internal/acdc_logdir/unet2D_ws_lr0/recursion_0_data.hdf5','r')

slice_idx = 480
image = base_file['images_train'][slice_idx, ...]
mask_gt = base_file['masks_train'][slice_idx, ...]
mask_pred = data_file['postprocessed'][slice_idx, ...]


d = dcrf.DenseCRF2D(exp_config.image_size[0], exp_config.image_size[1], exp_config.nlabels - 1)
labels = np.unique(mask_pred)
labels = labels[labels != 0]
U = unary_from_labels(labels,
                      n_labels=exp_config.nlabels - 1,
                      gt_prob=0.7,
                      zero_unsure=True)
d.setUnaryEnergy(U)

Q_unary = d.inference(10)

map_soln_unary = np.argmax(Q_unary, axis=0)
map_soln_unary =map_soln_unary.reshape(exp_config.image_size)

#d.addPairwiseGaussian(sxy=3, compat=3)







CRF = map_soln_unary[:]

fig = plt.Figure()

#Add image
ax = fig.add_subplot(141)
ax.axis('off')
ax.imshow(image, cmap='gray')

#Add gt
ax = fig.add_subplot(142)
ax.axis('off')
ax.imshow(mask_gt, cmap='jet', vmin=0, vmax=4)

#Add prediction
ax = fig.add_subplot(143)
ax.axis('off')
ax.imshow(mask_pred, cmap='jet', vmin=0, vmax=4)

#Add CRF
ax = fig.add_subplot(144)
ax.axis('off')
ax.imshow(CRF, cmap='jet', vmin=0, vmax=4)