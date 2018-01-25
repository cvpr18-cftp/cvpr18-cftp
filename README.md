# Coarse-to-Fine Localization of Temporal Action Proposals (CFTP)
This repository holds the code and models for CFTP. The code contains (1) training network, test network and sovler files .prototxt for the CPN, CAN and PRN metioned in the paper; (2) the tools of training and validation data building for three actionness learning in CPN; The code in this repository needs 3rd dependency:
- caffe (https://github.com/BVLC/caffe)


# Contents
* [Get the code](#get-the-code-back-to-top)
* [Training/extracting multiple actionness curves (CPN)](#trainingextracting-multiple-actionness-curves-cpn-back-to-top)
  * [Build training/validation dataset](#build-trainingvalidation-dataset-back-to-top)
    * [Steps](#steps-back-to-top)
    * [Data format](#data-format-back-to-top)
  * [Train actionness curve](#train-actionness-curve-back-to-top)
  * [Extract actionness score](#extract-actionness-score-back-to-top)
* [Training/testing Convolutional Anchor Network (CAN)](#trainingtesting-convolutional-anchor-network-can-back-to-top)
* [Training/testing Proposal Reranking Network (PRN)](#trainingtesting-proposal-reranking-network-prn-back-to-top)
* [Notation](#notation-back-to-top)
  
----
## Get the code [[back to top](#coarse-to-fine-localization-of-temporal-action-proposals-cftp)]
Use git to clone this repository and its submodules
```
git clone --recursive https://github.com/cvpr18-cftp/cvpr18-cftp
```

## Training/extracting multiple actionness curves (CPN) [[back to top](#coarse-to-fine-localization-of-temporal-action-proposals-cftp)]

### Build training/validation dataset [[back to top](#coarse-to-fine-localization-of-temporal-action-proposals-cftp)]

#### Steps [[back to top](#coarse-to-fine-localization-of-temporal-action-proposals-cftp)]

```
(1) Build the caffe repository.
(2) Add the "convert_frame_fea_point.cpp/convert_frame_fea_pair.cpp/convert_seq_fea_recurrent.cpp" into the project.
(3) Build the solution and get the point-wise/pair-wise/recurrent-wise actionness data construction tools.
``` 

#### Data format [[back to top](#coarse-to-fine-localization-of-temporal-action-proposals-cftp)]

The data construction tools need the video auxiliary information files and binary format features. Following are the format of input files for these tools:


| Input File                                                      |  Format                                                               |
|-----------------------------------------------------------------|-----------------------------------------------------------------------|
| VIDEO_NAME_LIST                                                 | each row: train/validation video_name                                 |
| BINARY_FEATURE (BIN_DIR + PREFIX.SBUFIX)                        | 1 int (D, e.g., 4) + D int (e.g., NCHW) + N\*C\*H\*W float (e.g. 1\*1024\*1\*1) |
| FEATURE_NAME_LIST (each row corresponding to one feature)       | each row: video_name \space frame_index \space actionness label (0/1) |

The extracted binary features on THUMOS14/ActivityNet (global_pool in BN-inception and pool5 in Pseudo-3D) will be released with the auxiliary files. 


### Train actionness curve [[back to top](#coarse-to-fine-localization-of-temporal-action-proposals-cftp)]


Use the training .prototxt and solver file in ./CPN/pointwise, ./CPN/pairwise and ./CPN/recurrent to train three actionness models.

### Extract actionness score [[back to top](#coarse-to-fine-localization-of-temporal-action-proposals-cftp)]


Use the testing .prototxt file in ./CPN/pointwise, ./CPN/pairwise and ./CPN/recurrent to extract three actionness curves for coarse proposal generation.


## Training/testing Convolutional Anchor Network (CAN) [[back to top](#coarse-to-fine-localization-of-temporal-action-proposals-cftp)]


Use the training and solver file in ./CAN to train CAN and use the test prototxt file to generate fine-grained proposals.


## Training/testing Proposal Reranking Network (PRN) [[back to top](#coarse-to-fine-localization-of-temporal-action-proposals-cftp)]



Use the training and solver file in ./PRNet to train PRN and use the test prototxt file to rerank proposals from previous stages.


## Notation [[back to top](#coarse-to-fine-localization-of-temporal-action-proposals-cftp)]


Please note that the codes for CAN and PRN training/testing data construction and data layer in caffe are not included in the current branch. The source code for these tools and layers will be released with our custom caffe.



