The start-of-the-art cardiac segmentation networks upon which this work builds, 
including the setup, can be found on [Christian Baumgartner's GitLab Page](https://git.ee.ethz.ch/baumgach/acdc_segmenter_internal)
Authors:
- Christian F. Baumgartner ([email](mailto:baumgartner@vision.ee.ethz.ch))
- Lisa M. Koch ([email](mailto:lisa.koch@inf.ethz.ch))
- Basil Mustafa ([email](mailto:bm490@cam.ac.uk))

Broadly, the following functionality was added:
- Recursively train using weakly supervised data
- Random walk before training and between recursions
- Algorithmically shorten scribble supervision
- Postprocess ground truths between recursion
- Debug training sequence (random walk / CNN / postprocess / random walk ...)
- Segment with variable number of labels as defined in experiment configuration
- Utility to save images (uses matplotlib - won't work on cluster)

## Requirements 
- Python 3.4 (only tested with 3.4.3)
- Tensorflow >= 1.0 (only tested with 1.1.0)
- The remainder of the requirements are given in `requirements.txt`
 

# Running the code

## Getting the code

Simply clone the repository by typing

``` git clone git@git.ee.ethz.ch:bmustafa/acdc_segmenter_internal.git ```

If this is your first time using the D-ITET gitlab server, you will need to [setup an SSH key](https://git.ee.ethz.ch/help/ssh/README.md) first.  

## Training the network

_Note: You will need the scribbled data file - see the [getting the data](##Getting the Data) section to set that up_

* Set up experiment configuration file (e.g. ```experiment_name.py``` in /experiments folder, changing parameters as necessary
    * do _not_ delete parameters - it will likely crash. Simply set to `False` or `None` to disable a postprocessing option
    * you may need to alter the data settings in the configuration to match where you place the hdf5 file
* Edit the line `from experiments import experiment_name as exp_config` at the top of ```train_weak.py```
* Run `python train_weak.py`

## Evaluate Patients
_Evalutes segmentation accuracy metrics on validation data set_

_Usage:_

```python evaluate_patients.py exp_path --save=1/0 --recursion=int --postprocess=1/0```

    Input Params:
        exp_path:       Not optional
                        Path, relative to project root folder, of experiment log directory
        
        --save:         Optional, 1/0.
                        Chooses whether or not to save images with best/worst/median dice as .png
                        If not specified, will not save images 
        
        --recursion     Optional, integer.
                        Chooses recursion to evaluate.
                        If not specified, will choose most recent.
                        
        --postprocess   Optional, 1/0.
                        Chooses whether to only keep largest cluster after segmentation
                        If not specified, will not postprocess.
   
## Debug Processing
_Shows progression of (2D) dice with random walking/postprocessing/recursion sequence_

_Usage:_

```python debug_processing.py exp_path --save=1/0 --num_images=int```

    Input Params:
        exp_path:       Not optional
                        Path, relative to project root folder, of experiment log directory
        
        --save:         Optional, 1/0.
                        Chooses whether or not to save csv file of dices.
                        If not specified, will not save.
        
        --num_images    Optional, int.
                        Choose number of (randomly sampled) images to show processing progression on
                        If not specified, will show best/median/worst scenario
                        
                        
## Get Images
_Script to create fully supervised and weakly supervised segmentations and save as png_

_Usage:_
```bash get_images.sh ws_exp_path fs_exp_path test_data(1/0) slice_1 slice_2 ... slice_n```

    Input Params:
        ws_exp_path:    Not optional
                        Path, relative to project root folder, of weakly supervised experiment log directory
                        
        fs_exp_path:    Not optional
                        Path, relative to project root folder, of weakly supervised experiment log directory
        
        test_data:      Not optional, 1/0.
                        Chooses whether to predict segmentations from testing or training data.
        
        slices          Sequence of integers of slice indices (e.g. 180 438 920)
                        Chooses which slices to process
                        
# Results & Future Work

## Achieved results
_Note: Experiments were rerun due to a bug with the final recursion data file getting corrupted, but have not been reevaluate, so these numbers may be slightly off but should broadly be the same_

To summarise:
* Recursion was only useful for more weakly supervised cases, but the effect of recursion
is conflated with the postprocessing options and this needs to be more carefully explored.
* Weak supervision can achieve results 96% as good as full supervision, and spot supervision
can achieve results 91% as good.

| **Experiment**                       |**Recursion**|**LV**|**Myo**|**RV**|**Mean Dice**|
|-------------------------------------:|:--------:|:----:|:----:|:----:|:--------:|
| Full supervision (paper)             |          | 0.950 | 0.899 | 0.893 | 0.914     |
|                                      |          |       |       |       |           |
| Weak supervision (no recursion)      |          | 0.925 | 0.873 | 0.839 | 0.879     |
|                                      |          |       |       |       |           |
| Weak supervision (recursion)         | 0        | 0.916 | 0.865 | 0.830 | 0.870     |
| _Postprocessing on_                  | 1        | 0.914 | 0.860 | 0.824 | 0.866     |
|                                      | 2        | 0.910 | 0.861 | 0.816 | 0.862     |
|                                      |          |       |       |       |           |
| Weak supervision                     | 0        | 0.711 | 0.584 | 0.542 | 0.612     |
| _No random walk_                     | 4        | 0.592 | 0.646 | 0.510 | 0.583     |
|                                      |          |       |       |       |           |
| Weak: 50% scribble length            | 0        | 0.890 | 0.770 | 0.790 | 0.817     |
| _Postprocessing on_                  | 1        | 0.890 | 0.792 | 0.815 | 0.832     |
|                                      | 2        | 0.895 | 0.788 | 0.815 | 0.832     |
|                                      | 3        | 0.899 | 0.793 | 0.814 | 0.835     |
|                                      |          |       |       |       |           |
| Weak: 50% scribble length            | 0        | 0.911 | 0.800 | 0.816 | 0.842     |
| _Postprocessing off_                 | 1        | 0.903 | 0.791 | 0.821 | 0.839     |
|                                      | 2        | 0.894 | 0.807 | 0.818 | 0.840     |
|                                      |          |       |       |       |           |
|                                      | 0        | 0.859 | 0.592 | 0.708 | 0.720     |
| Weak: Spot supervision               | 1        | 0.898 | 0.766 | 0.796 | 0.820     |
| _Postprocessing on_                  | 2        | 0.885 | 0.807 | 0.790 | 0.828     |
| _RWI + RWR_                          |          |       |       |       |           |
|                                      |          |       |       |       |           |
| Weak: Spot supervision               | 0        | 0.896 | 0.591 | 0.735 | 0.740     |
| _KLC + RWR_                          | 1        | 0.892 | 0.617 | 0.772 | 0.760     |
|                                      | 2        | 0.884 | 0.606 | 0.752 | 0.747     |
|                                      |          |       |       |       |           |
| Weak: Spot supervision               | 0        | 0.888 | 0.598 | 0.740 | 0.742     |
| _Postprocessing off_                 | 1        | 0.886 | 0.641 | 0.764 | 0.763     |
|                                      |          |       |       |       |           |
|                                      |          |       |       |       |           |
| Mixed - 25% Full Supervision         | 0        | 0.936 | 0.875 | 0.875 | 0.896     |
| _Postprocessing off_                 | 1        | 0.933 | 0.836 | 0.846 | 0.872     |
|                                      | 2        | 0.933 | 0.833 | 0.835 | 0.867     |

## Future Work
### Implementation
* Postprocessing which _refines_ the segmentation
    * Currently, random walker doesn't act to refine the segmentation - it helps bridge the gap between weak and
      full supervision. Seeing as most inaccuracies occur at the edges of structures, edge based approaches (e.g.
      CRFs) are suggested for future exploration
* Optimisation of random walker parameters (algorithmically or otherwise)

### Experimentation
* More robust experiments to evaluate the impact of recursion
    * For spot supervision, could similar results just be achieved by initially using a fully random walked segmentation as the
ground truth? Currently it uses thresholded random walk as the ground truth, starts to learn a half-decent segmentation and then, due to the
_random walker intersection_ postprocessing the ground truth basically becomes the fully random walked segmentation and the network starts to
produce much better segmentations.

## Getting the data

Currently the data I scribbled on is in `/itet-stor/bmustafa/amgen`

