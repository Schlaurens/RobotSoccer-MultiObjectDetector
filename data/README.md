## Overview of data directories

#### `./groundtruth`
- Contains the groundtruth data
- All directories contain:
  - a `.jpg` image for each sample
  - a `labels.json` that contains the following attributes for each sample:
    - `name`: the name of the corresponding image
    - `frame_time`: the timestamp of the current sample.
    - `cpose`: the camera pose
    - `cintr`: the intrinsic camera parameters
    - `intersections`: image coordinates for L-, T-, X-Intersections if they exist in this sample. And an `ignore_sample` flag that is true if this sample is to be ignored in the training loss.
    - `penaltyMark`: image coordinates of the penaltyMark it is exists in this sample. 
    - `ball`: image coordinates and `radius` (in cm) of the ball if it exists in this sample.
- Also contains `.tfrecords` files for each directory. These files contain all groundtruth data used for training:
  - `name`: The unique identifier (`String`) of the current sample in the dataset. (Name of the logfile + _ + Image name).
  - `frame_time`: The time stamp (`int32`) od the current sample.
  - `image`: a `tf.Tensor` of shape `[480, 320, 4]` containing the image.
  - `camera`: a `tf.Tensor` of shape `[3]` containing the camera pose.
  - `intrinsics`: a `tf.Tensor` of shape `[4]` containing the intrinsic camera parameters.
  - `ball`:
    - `object_mask`: a `tf.Tensor` of shape `[15, 20]` containing a binary mask that indicates in which cell the ball is.
    - `offset_mask`: a `tf.Tensor` of shape `[15, 20, 2]` containing the offset from the center of each cell to the ball.
    - `loss_mask`: a `tf.Tensor` of shape `[15, 20]` containing a binary mask that indicates which cell should have an impact on the training loss.
  - `penaltyMark`:
    - `object_mask`: a `tf.Tensor` of shape `[15, 20]` containing a binary mask that indicates in which cell the penaltyMark is.
    - `offset_mask`: a `tf.Tensor` of shape `[15, 20, 2]` containing the offset from the center of each cell to the penaltyMark.
    - `loss_mask`: a `tf.Tensor` of shape `[15, 20]` containing a binary mask that indicates which cell should have an impact on the training loss.
  - `intersections`:
    - `object_mask`: a `tf.Tensor` of shape `[15, 20]` containing a binary mask that indicates in which cell an intersection is.
    - `offset_mask`: a `tf.Tensor` of shape `[15, 20, 2]` containing the offset from the center of each cell to the **nearest** intersection. If two intersection are in the same cell, the lower one is taken.
    - `loss_mask`: a `tf.Tensor` of shape `[15, 20]` containing a binary mask that indicates which cell should have an impact on the training loss.
    - `classification_mask`: a `tf.Tensor` of shape `[15, 20]` containing the class of the intersection in the cell. The class is indicated by a number: `0 -> None`, `1 -> L`, `2 -> T`, `3 -> X`.


#### `./tfrecords`
- Contains the complete datasets in `.tfrecords` format.
- These datasets are split into train-, validation-, test-data.
- The dataset files contain the shuffled contents of the `.tfrecords` file from `./groundtruth`.

#### `./prelabeled`
- Contains the data as it was extacted from the B-Human gamelogs. This includes images, camera pose, intrinisics camera parameters as labels for intersections, balls and penaltyMarks as they were logged.

#### `./b-human_predictions`
- Contains a `.json` file for each game log that contains the labels of intersections, balls and penaltyMarks that were predicted by the current B-Human object detectors.

#### `./evaluation`
- Contains the predictions for the test samples in a single `.json` file for the B-Human predictions, the model predictions and the groundtruth.
- The selected samples for which predictions were made are the same samples inside test dataset from the `./tfrecords` directory.
- These files are used to compare the model performance with the current B-Human predictors (as of 27.1.2026)