
---

**Vehicle Detection Project**


[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example_car.png
[image3]: ./output_images/HOG_example_not_car.png
[image4]: ./output_images/sliding_windows.jpg
[image5]: ./output_images/sliding_window.png
[image6]: ./output_images/bboxes_and_heat.png
[image7]: ./output_images/labels_map.png
[image8]: ./output_images/output_bboxes.png
[video1]: ./output_images/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 97 through 169 of the file called `utils.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images from the KITTI, GTI datasets and extracted images from the Udacity crowd-ai dataset. 
In order to avoid time-series data (specifically in the GTI set) I selected randomly a third of vehicle image for a maximum of about 9000 vehicle images. I did not use this random selection approach for the non-vehicle images. The training set was balanced with both classes having the same number of images.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:



![alt text][image2]
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the above parameters produced the best accuracy results when training the classifier.
In particular the LUV color space appeared well suited for detecting white colors and was an improvement over RGB color space. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using, HOG features, color histogram and spatial binning. The spatial size selected to work well was (20,20) and the number of histogram bins for the color histogram was 128. Increasing the spatial size above (20,20) produced marginally better result but increased the size of the feature vector, which was not computationally practical. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I searched each frame with a sliding window of size (64,64) pixels. I further used the following scales: 1, 1.25, 1.5, 1.75, 2, 2.5, which allowed detection of vehicles with varying size between 64x64 pixels and  160x160 pixels. Vehicles further away could be possibly be detected with scales below 1, but this was not explored. 

Here is an example of an image with detections at different scales:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on six scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. The classifier test accuracy was 98.7% with training accuracy over 99%. I mitigated against overfitting by adjusting the C parameter of the SVM. I further used a svc.decision_function with a threshold of 0.2 in order to obrain only detections where the classifier has higher confidence. This was done to avoid false positives.

Here are some example images:

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I also implemented a filter integrating the heatmaps of the last ten images and then thresholded at a certain level to smooth out the detection and avoid false positives. 

To further overcome problems with false positives I also recorded all false positives and fed them back to the classifier (hard negative mining). This reduced the false positives to a minimum (single digit false positives relative to more than 24,000 detections). The algorithm is amazingly even able to detect vehicles behind the crash barriers, where only part of the roof of the vehicle is visible.

I further used svc.decision_function to only accept detections where the classifier outputs distance from the decision boundary greater than 0.2. This was done to avoid false positives, that can be close to the decision boundary.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a frame and its corresponding heatmap:

![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![alt text][image7]

### Here the resulting bounding boxes are drawn onto a frame in the series:
![alt text][image8]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? 

I experimented with various parameters (from the number and type of images, to HOG and classifier parameters). I limited the area in the image where detections can occur in order to speed up the algorithm and also varied the size of the area depending on the scale at which detection can occur (vehicles further away can be detected only at smaller scales, while those near can be detected with larger scales).
I also implemented an identification and tracking of individual vehicles. Each new detection then is allocated to previously detected cars if it overlaps with the previously detected bounding box of the car (using Intersection over Union algorithm). Then to present a smooth bounding box around the car, I averaged the bounding box for the individual car over the last n detections. 
I further used hard-negative mining to reduce false positive detections.

The project was challenging, not least because of the number of free variables that needed to be adjusted manually. While adjusting all parameters manually lead to a satisfactory result, there remains the question of robustness. A real environment rarely adheres to a large set of manually selected parameters. Another major issue with the algorithm is the speed of detection. Every second in the video took between 30 seconds and 1 minute to process, making the algorithm unusable in practice. Still, the project shows that it is possible (although perhaps not practical) to detect vehicles with a more manual approach as opposed to a convolutional neural network. 
