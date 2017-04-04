##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/perspective_straight_lines1.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./output_images/calibration_example.png "Calibration Example"
[image8]: ./output_images/undistorted_example.png "Undistorted Example"
[image9]: ./output_images/threshold_example.png "Threshold Example"
[image10]: ./output_images/RCurve.png "Curvature Equation"
[image11]: ./output_images/P4_VideoScreenShot.png "Output Example"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

To generate video use P4Video.py as follows:
P4Video.py <input_file> <output_file>  

---
###Writeup / README

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is called in lines 31 through 43 of the file called `P4.py` and uses functions defined in lines 8 through 52 (funcions: `camera_setup` and `camera_calibrate`) of the file called Helper-Functions.py).

I start by preparing "object points", objp, which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am applying planar homography, i.e. assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time, assuming I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image7]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image8]

The output from `cv2.calibrateCamera()` includes camera matrix, `mtx` and distortion coefficients `dist`. The code for function to undistort is contained in lines 54 through 67 (function: `cal_undistort`) of the `Helper_Function.py` file. Given an original image, camera matrix and distortion coefficient, this function outputs the undistorted image using `cv.undistort` function.

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 176 through 179 (function:`pipeline`) of `Helper_Function.py`).  The color threshold only keeps pixels with `s` values between 170 and 255, after conversion to the HSV color space (lines 162 through 171 (function:`color_threshold`) of `Helper_Function.py`). The gradient threshold only keeps pixels with an absolute gradient between 20 and 100 in the x-direction or y-direction after conversion to grayscale and using Sobel (lines 107 through 124 (function:`abs_sobel_thresh`) of `Helper_Function.py`). Here's an example of my output for this step.

![alt text][image9]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform directly calls `cv2.warpPerspective` (appears in line 148 in the file `P4.py`), which takes as input parameter the perspective transform previously calculated. The function to compute the perspective transform for the warp and its inverse is the function `get_perspective_transform()` which takes as inputs an image (`img`), as well as optional source (`src`) and destination (`dst`) points. If not provided, the default source and destination points in the following manner (experimentally found by using the given two straight_line examples):

```
src = np.float32(
    [[(img_size[1] / 2) - 60, img_size[0] / 2 + 100],
    [((img_size[1] / 6) - 10), img_size[0]],
    [(img_size[1] * 5 / 6) + 50, img_size[0]],
    [(img_size[1] / 2 + 65), img_size[0] / 2 + 100]])
dst = np.float32(
    [[(img_size[1] / 4), 0],
    [(img_size[1] / 4), img_size[0]],
    [(img_size[1] * 3 / 4), img_size[0]],
    [(img_size[1] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1117, 720     | 960, 720      |
| 705, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Lane-line pixel identification and polynomial fit (lines 243 through 398 (functions: `get_lane_lines` and `get_lane_lines_with_prior`) in my code in `Helper_function.py`)
There are 2 functions that are used for lane-line pixel identification and polynomial fit. `get_lane_lines_with_prior` uses input prior left and right lanes to initialize the lane-line search while `get_lane_lines` is called if there aren't any prior lines to reference. Instead this function uses a histogram and sliding windows in a more tedious search. 

After obtaining candidate pixels for the left and right lane for I found my lane lines by fitting a 2nd order polynomial using numpy's polyfit, pretty much exactly like the project lesson.

####5. Radius of curvature and vehicle position (lines 483 through 505 (functions: `get_curvature`) in my code in `Helper_function.py`)

For radius of curvature I used the sample implementation provided in the lession to compute the radius of curvature according to this equation where A, B, C are the coefficients of the 2nd order polynomial equation computed from 4.

![alt text][image10]

For vehicle position I compared the center of the image to the center of the left and right detected lanes.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 119 through 137 (function: `draw`) of my code in `P4.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image11]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This implementation works well for the majority of the project video. Where it would fail is the lack of contrast of lane lines or creation of false-positive lines from shadows or road coloring. 

To improve robustness I would start with better color transforms and tresholds to find edge candidates for lane lines. Currently the conversion to grayscale for gradient equally considers all channels but a better approach may be to weight channels differently and to dynamically adjust these weights depending on the reliability of the previous frame. 

In addition kernels used best detects vertical and horizontal lines while are targets are curves. We can adjust the kernel for curves based on angle computation. This has the potential to improve in the harder challenge video when the road curvature is more frequent and amplitudes are higher. 

