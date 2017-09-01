import model_zoo
import tensorflow as tf

experiment_name = 'unet2D_ws_25fs'

# Model settings
model_handle = model_zoo.unet2D_bn_modified

# Data settings
data_mode = '2D'  # 2D or 3D
image_size = (212, 212)
target_resolution = (1.36719, 1.36719)
scribble_data = '/scribble_data/scribbled_data.hdf5'        #Path from project root

# Training settings
batch_size = 10
learning_rate = 0.01
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = True
weight_decay = 0.00000
momentum = None
loss_type = 'crossentropy_incomplete'  # crossentropy/weighted_crossentropy/dice

# Augmentation settings
augment_batch = False
do_rotations = True
do_scaleaug = False
do_fliplr = False

# Rarely changed settings
use_data_fraction = False  # Should normally be False
nlabels = 5
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                                       # be reduced

train_eval_frequency = 200
val_eval_frequency = 100

# Weak supervision settings
random_walk = True
rw_beta = 1000
rw_threshold = 0.99
epochs_per_recursion = 50
number_of_recursions = 4

#CNN Options
reinit = False                  # if true, will reinitialise network weights between recursions
cnn_threshold = 0.8              # if defined, will threshold output of CNN such that more unlabelled pixels are present
use_crf = True

#Postprocessing options
keep_largest_cluster = False    # if true, will only keep the largest cluster in the output
                                # (therefore more space for random walker + recursion to do work)
rw_intersection = True          # if true, will random walk to fully segment image based off original scribble
                                # then limit output to the regions defined by the low threshold random walk
rw_reversion = True             # if true, will attempt to revert to original random walked scribbles if the
                                # if CNN + postprocessing predicts a smaller structure than was in the original
                                # scribble
                                #Parameters for gaussian edge smoothing
                                #Larger sigma blurs more, smaller threshold results in more growth
edge_smoother_sigma = None      #
edge_smoother_threshold = None  # between 0 & 1

percent_full_sup = 25
length_ratio = 1          # factor by which to reduce the length of the scribbles

#AUTOCALCULATED
postprocessing = bool(reinit + keep_largest_cluster + bool(cnn_threshold) + rw_intersection + rw_reversion)
max_epochs = number_of_recursions*epochs_per_recursion
smooth_edges = (not edge_smoother_sigma is None) and (not edge_smoother_threshold is None)
