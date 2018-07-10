# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  


The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  



[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image4]: ./output_images/img_box.png
[image5]: ./output_images/heatmap.png
[image6]: ./output_images/labeled.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4



---
### Writeup / README



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second and third code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and check it by validating the trained SVM. Finally, I chose `YCrCb` as the color space and `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the following steps:

* Read the data
* Extracted the features of the image include:
  * Histogram of Oriented Gradients (HOG)
  * Color features (spatial and histograms)
* Normalized the data
* Randomized the order of the data
* Split the data into train and test parts for validation

Finally, the model reached an accuracy of 99.3%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search limited window positions at five different scales in the image:

Window size| Y start | Y end| X start| Overlap ratio
-----|-----|---|---|---
70x70|  380|490|606|0.8
90x90|  380|530|476|0.7
110x110|380|570|346|0.6
130x130|380|610|216|0.5
150x150|380|650| 86|0.4


Here is the code of `multi_window_search` function:

        winsow_sizes = (70, 90, 110, 130, 150)
        def multi_window_search(image, winsow_sizes, svc, X_scaler, color_space='RGB',
                            spatial_size=(32, 32), hist_bins=32,
                            hist_range=(0, 256), orient=11,
                            pix_per_cell=11, cell_per_block=2,
                            hog_channel=0, spatial_feat=True,
                            hist_feat=True, hog_feat=True):
            hot_windows=[]
            for i in range(len(winsow_sizes)):
                xy_window = (winsow_sizes[i], winsow_sizes[i])
                y_start = 380
                y_stop = 490+40*i
                x_start = 606-i*130
                overlap = 0.8 - 0.1*i
                y_start_stop = [y_start, y_stop]
                windows = slide_window(image, x_start_stop=[x_start, None], y_start_stop=y_start_stop,
                                    xy_window=xy_window, xy_overlap=(overlap, overlap))
                hot_window = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)        
                hot_windows.extend(hot_window)
            return hot_windows

        hot_windows = multi_window_search(image.astype(np.float32)/255, winsow_sizes, svc, X_scaler, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)   

        draw_image = np.copy(image)
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

        plt.imshow(window_img)


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using `YCrCb` 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  I also optimized the searching area of the window sliding method and the size of the windows to get a better result.
Here are some example images:

###### Original image with the boxes
![alt text][image4]

###### Heatmap
![alt text][image5]

###### Labeled image
![alt text][image6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the heatmap stream and calculate the average heatmap of the last four frames to get a stable and accurate result adopted 1.5 as the final threshold. In that case, the true position of the vehicles would be enhanced while the influence of the outliers would be weakened due to the average calculation. Notice that the heatmap recorded is the data before applying the threshold but not the processed one. Here is the code of the pipeline:

    hot_cache=[]
    frame_num=0
    cache_num = 4
    def img_proc(image):
      global hot_cache
      global frame_num
      if frame_num > cache_num:
          hot_windows = multi_window_search(image.astype(np.float32)/255, winsow_sizes, svc, X_scaler, color_space=color_space,
                              spatial_size=spatial_size, hist_bins=hist_bins,
                              orient=orient, pix_per_cell=pix_per_cell,
                              cell_per_block=cell_per_block,
                              hog_channel=hog_channel, spatial_feat=spatial_feat,
                              hist_feat=hist_feat, hog_feat=hog_feat)  
          heat = np.zeros_like(image[:,:,0]).astype(np.float)
          heat = add_heat(heat, hot_windows)
          hot_cache.append(heat)
          hot_cache = hot_cache[-cache_num:]
          heat_best = np.average(hot_cache, axis=0)
          heat = apply_threshold(heat_best, 1)
          # Visualize the heatmap when displaying    
          heatmap = np.clip(heat, 0, 255)
          # Find final boxes from heatmap using label function
          labels = label(heatmap)
          draw_img = draw_labeled_bboxes(np.copy(image), labels)
          frame_num += 1
      else:
          hot_windows = multi_window_search(image.astype(np.float32)/255, winsow_sizes, svc, X_scaler, color_space=color_space,
                      spatial_size=spatial_size, hist_bins=hist_bins,
                      orient=orient, pix_per_cell=pix_per_cell,
                      cell_per_block=cell_per_block,
                      hog_channel=hog_channel, spatial_feat=spatial_feat,
                      hist_feat=hist_feat, hog_feat=hog_feat)  
          heat = np.zeros_like(image[:,:,0]).astype(np.float)
          heat = add_heat(heat, hot_windows)
          hot_cache.append(heat)
          heat = apply_threshold(heat, 1)
          heatmap = np.clip(heat, 0, 255)
          # Find final boxes from heatmap using label function
          labels = label(heatmap)
          draw_img = draw_labeled_bboxes(np.copy(image), labels)
          frame_num += 1
      return draw_img


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The approach worked well in the given video and it also identified the cars coming from the other side of the road. Though it was able to find out the car in the image, the location and the size of the block is still not precise enough to calculate the distance of the car from the observer and the real size of the detected car. The approach still needs to be modified to get an accurate and precise result under pixel level. During the test I found that dense windows would help making the result more accurate and robust but required massive computation which significantly slowed down the detection speed. We may apply the deep learning method to increase accuracy and precision of the result such as `SSD`.

Another problem was that the approach was not able to separate the two cars if they were close to each other in the image.  I am wondering if we can use stereo imaging as the input source and the new depth information may help identify the two different cars during searching. Deep learning can be another solution and we can add such case in the training dataset to bring the model more knowledge of segmentation.
