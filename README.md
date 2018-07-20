# Vehicle Detection Project

[//]: # (Image References)
[hog-vis]: ./examples/hog-visualization.png 'HOG Visualization'
[find_cars]: ./examples/find_cars.png 'Window Search Analysis'
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

### Histogram of Oriented Gradients (HOG) Visualizations

Visualized a comparision between non-vehicle and vehicle images of extracted HOG
samples for each of the popular color spaces.

- YCrCb
- RGB
- HSV
- HLS
- YUV
- LUV

Here is an example of `YCrCb`

![alt text][hog-vis]


After reviewing all of the visualizations of HOG samples from the diferent color spaces 2 seemed to stand out: `YUV`
and `YCrCb`.  

To further validate the color space the normalized features were tested on a Linear SVM. `YCrCb` seemed to have the
best validation accuracy on a consistent basis. 

#### Training the Classifier

Decided to use a Linear SVM as suggested in the lesson for it's blend of speed and accuracy on this problem. 
An attempt to use a non-linear kernel in the SVM was also tried with slightly better accuracy and less false positives
but performance suffered dearly.

To tune the parameters I used a combination of a `GridSearchCV` search and testing accuracy on the validation set.

From my testing to get the best accuracy I used a normalized feature vector a `color historgram`, `spatial binning` and 3 channel 
`HOG` extractions from the `YCrCb` color space.

### Sliding Window Search

Used sub-sampling for each window in the search. To get different sample sizes I resized the entire region of interest
(as demonstrated in the lesson) prior to applying the sliding window. The classifier seemed to be picky on size so I ran
the project with 2 different scales: `1.0` and `1.5`.

Because of the overlapping and the multiple scales used to find the vehicles there were many duplicates and partial matches.
I used a heatmap along with scipy's `label` function to find bounding box of each hot region.

Some of the frames created had false positives which could be filtered via the heatmap with a tunable threshold and setting
a region of interest to avoid detection of cars where they should never be.

![alt text][find_cars]
---

### Video Pipeline

Here's a [link to my video result](./drawn_project_video.mp4)

First step was to create a video pipeline that was completely stateless (no knowledge of previous frames). That created
a lot of false positives even with a balanced threshold. 

To cope with this, I made the heatmap stateful to take in a max_size where old heatmaps have a TTL and will fall off.
When the heatmap is needed an aggregation is applied to the recent values. 

Aggregations applied to the stored recent heatmaps was does this pixel occur in multiple heatmaps and takes an average
of the heatmaps with a threshold.

### Discussion

The trickiest part of the project is cleaning up the false positives. I played a lot with thresholding heatmaps,
different color spaces and different SVM parameters. Using a non-linear classifier showed better results but was very
slow. It seems to come down to processing over multiple frames to get those values that happen more consistently.

A future enhancement I'd like to try applying a Kalman Filter and tracking each vehicle.

If I were to solve this problem with less direction I'd try to apply more deep learning techniques for object detection.