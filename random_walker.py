__author__ = 'bmustafa'
import numpy as np
import warnings
from skimage.segmentation import random_walker
from skimage import morphology, measure
from scipy.ndimage.morphology import binary_dilation
from scipy.signal import convolve2d
from math import ceil


def segment(image, seeds, threshold=0.95, beta=90, bg_label=-1, return_bg_label=True, debug=False):
    slice_count = seeds.shape[0]
    #Labels are unique values in the mask
    labels = np.unique(seeds).astype(int)
    nlabels = len(labels)
    #Remove zero label
    labels = np.delete(labels, 0)

    #Cast seed array to integer
    seeds = seeds.astype(int)

    #Squeeze inputs
    image = np.squeeze(image)
    seeds = np.squeeze(seeds)

    #Initialise array of new labels
    new_labels = np.zeros(image.shape)

    #Handle background label
    if bg_label == -1:
        #Assume largest label is the background
        bg_label = np.amax(labels)
    elif bg_label == 0:
        #Warn for no background label
        warnings.warn("WARNING: Input does not have labelled background. "
                      "Results are typically much better with background seeds.")
    for sliceNo in range(0, slice_count):
        #isolate image data + seeds for this slice
        data = np.squeeze(image[sliceNo, :, :])
        markers = np.squeeze(seeds[sliceNo, :, :])

        #Normalise data
        data -= np.min(np.min(data))
        data /= 0.5*np.max(np.max(data))
        data -= 1

        labels = np.unique(markers)
        if 0 in labels:
            labels = np.sort(labels[labels!=0])
            try:
                probs = random_walker(data, markers, beta=beta, mode='bf', return_full_prob=True)
                #First, wherever the probability of other labels is higher, set label probability to zero
                #One hot matrix for all labels
                p_all = np.zeros([probs.shape[1], probs.shape[2], nlabels])
                p_all[..., 0] = np.ones([probs.shape[1], probs.shape[2]])*threshold
                ind = 0
                for label in labels:
                    p_all[..., label] = probs[ind, ...]
                    ind+= 1

                mask_out = np.argmax(p_all, -1).astype(dtype=np.int)
                new_labels[sliceNo, :, :] = mask_out.astype(dtype=np.int)[:]
            except Exception:
                warnings.warn("WARNING: Error computing segmentation for slice {}. Outputting original seeds.".format(sliceNo))
                new_labels[sliceNo, :, :] = markers
        else:
            new_labels[sliceNo, :, :] = markers
    return new_labels

def reduce_scribble_length(scribbles, ratio, minimum=10, keep_closed_loops=False, method='circular_mask'):
    def keep_max_connectivity(skeleton):
        #replace each pixel in the skeleton with it's connectivity
        skel = skeleton[:].astype(np.int)
        initial_pixel_count = np.sum(skel)
        connectivity = skeleton[:].astype(np.int)
        for [i, j], val in np.ndenumerate(connectivity):
            if val != 0:
                connectivity[i,j] = np.sum(skel[i - 1:i + 2, j-2: j + 1])
        #connectivity = convolve2d(skeleton.astype(np.int), np.ones([3,3]), mode='same').astype(np.int)
        maxval = np.max(connectivity)

        connectivity[connectivity != maxval] = 0
        connectivity[connectivity != 0] = 1

        if np.sum(connectivity) < minimum:
            return connectivity, False
        else:
            if np.sum(connectivity) == initial_pixel_count:
                #Can't reduce further
                return connectivity, False
            else:
                return connectivity, True

    def reduce(scribble):
    #get reduction of scribble
        skeleton = morphology.skeletonize(scribble)
        reduce_further = True if np.sum(skeleton) > 5 else False
        n = 0
        while reduce_further:
            skeleton, reduce_further = keep_max_connectivity(skeleton)
            n+=1
        #get indices of nonzero pixels
        if method == 'circular_mask':
            #need single central coordinate
            ind = np.transpose(np.nonzero(skeleton))
            if ind.shape[0] == 1:
                return ind.reshape(2)
            else:
                return np.average(ind, axis=0).astype(int).reshape(2)
        else:
            return skeleton

    def circular_mask(array, centre, radius):
        c_x, c_y = centre.reshape(2)
        w, h = array.shape
        Y, X = np.ogrid[:h, :w]
        centre_dist = np.sqrt((X - c_x)**2 + (Y - c_y)**2)
        mask = centre_dist < radius
        output = np.zeros_like(array)
        output[mask] = array[mask]
        return output

    if ratio < 0 or ratio >= 1:
        #return original scribbles if it's outside range
        return scribbles
    else:
        #get nonzero labels in image
        labels = np.unique(scribbles)
        #if there is only background label, return
        if len(labels) == 1 and labels[0] == 0:
            return scribbles
        else:
            labels = labels[np.nonzero(labels)]
            reduced_scribbles = np.zeros_like(scribbles)

            for label in labels:
                label_map = (scribbles.astype(np.int) == label)
                #seperate individual scribbles
                components = morphology.label(label_map, connectivity=2)
                num_components = len(np.unique(components)) - 1

                # print("In label {}, {} components found".format(label, num_components))

                for comp_idx in range(1, num_components + 1):
                    current_scribble = (components == comp_idx)
                    keep_old = False
                    if keep_closed_loops:
                        contours = measure.find_contours(current_scribble, 0.8)
                        if len(contours) != 1:
                            keep_old = True

                    if keep_old:
                        reduced_scribbles[current_scribble] = label
                    else:
                        reduction = reduce(current_scribble)

                        original_length = np.sum(current_scribble)

                        if original_length < minimum:
                            reduced_scribbles[current_scribble] = label
                        else:
                            target_length = max(np.sum(current_scribble)*ratio, minimum)

                            #try initial radius, then refine
                            r = 1
                            radial_increment = 1 if method == 'circular_mask' else max(ceil(target_length/20), 1)
                            current_length = 0
                            while current_length < target_length:
                                if method == 'circular_mask':
                                    seed_grown = circular_mask(current_scribble, reduction, r)
                                else:
                                    seed_grown = binary_dilation(reduction,
                                                                 structure=morphology.disk(radius=r))
                                    seed_grown[current_scribble == 0] = 0

                                current_length = np.sum(seed_grown)
                                r += radial_increment

                            reduced_scribbles[seed_grown] = label


            return reduced_scribbles