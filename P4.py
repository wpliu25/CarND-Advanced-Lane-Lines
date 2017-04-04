#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from Helper_Functions import *
import re

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # saveNext flag
        self.saveNext = True
        # was the line detected in the last iteration?
        self.detected = False
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #difference in fit coefficients between last and new fits
        self.diffs_left = np.array([0,0,0], dtype='float')
        self.diffs_right = np.array([0, 0, 0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        # Calibrations
        cam_cal = 'cam_cal.npz'
        warp_param = 'warp_param.npz'
        if not os.path.isfile(cam_cal):
            print('Calibrating camera...')
            self.mtx, self.dist = camera_setup()
            np.savez(cam_cal, mtx=self.mtx, dist=self.dist)
        else:
            print('Loading calibration from %s...' % cam_cal)
            data = np.load(cam_cal)
            self.mtx = data['mtx']
            self.dist = data['dist']
            print('Loading warp params from %s...' % warp_param)
            warp_data = np.load(warp_param)
            self.src = warp_data['src']
            self.dst = warp_data['dst']
            self.warp_m = warp_data['warp_m']
            self.warp_minv = warp_data['warp_minv']

        # x values of the last n fits of the line
        #self.recent_xfitted = []
        self.left_fit = np.array([0,0,0], dtype='float')
        self.right_fit = np.array([0,0,0], dtype='float')
        self.left_fitx = None
        self.right_fitx = None
        self.ploty = None

        #radius of curvature of the line in some units
        #self.radius_of_curvature = None
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        self.left_curverad = 0 # meters
        self.right_curverad = 0 # meters

        #distance in meters of vehicle center from the line
        self.line_base_pos = 0

    def check_lane_curvature(self, R, left=1):
        # Compare new curvature `R` to previous

        _r = self.right_curverad
        if(left == 1):
            _r = self.left_curverad

        # Return true if there is no prior data
        if _r is None:
            return True

        self.check = abs(R - _r) / _r

        return self.check <= 2/3

    def checkDetected(self, _left_fit, _right_fit, _left_fitx, _right_fitx, _ploty, _left_curverad, _right_curverad):
        if(not self.saveNext):
            if(self.check_lane_curvature(_left_curverad, 1) and self.check_lane_curvature(_right_curverad, 0)):
                self.detected = True
                self.left_fit = _left_fit
                self.right_fit = _right_fit
                self.left_fitx = _left_fitx
                self.right_fitx = _right_fitx
                self.left_curverad = _left_curverad
                self.right_curverad = _right_curverad
                self.ploty = _ploty
            else:
                self.detected = False
                #print("Left before: %.2f Now: %.2f" % (self.left_curverad, _left_curverad))
                #print("Right before: %.2f Now: %.2f" %(self.right_curverad, _right_curverad))
                #print("Not Detected Left!(%.2f,%.2f,%.2f)" % (self.diffs_left[0], self.diffs_left[1], self.diffs_left[2]))
                #print("Not Detected Right!(%.2f,%.2f,%.2f)" % (self.diffs_right[0], self.diffs_right[1], self.diffs_right[2]))
                self.saveNext = True
        else:
            self.detected = True
            self.left_fit = _left_fit
            self.right_fit = _right_fit
            self.left_fitx = _left_fitx
            self.right_fitx = _right_fitx
            self.left_curverad = _left_curverad
            self.right_curverad = _right_curverad
            self.ploty = _ploty
            self.saveNext = False

def AdvancedLaneLines(image, line, debug=0):

    # undistort image using previous camera calibration
    undist = cal_undistort(image, line.mtx, line.dist)

    # apply color and gradient thresholds
    color, threshold = pipeline(undist)

    # warp using previous parameters
    warped = cv2.warpPerspective(threshold, line.warp_m, (threshold.shape[1], threshold.shape[0]), flags=cv2.INTER_LINEAR)

    # determine lane lines using prior frame initialization
    _left_fit = None
    _right_fit = None
    _left_fitx = None
    _right_fitx = None
    _ploty = None
    _out_img = None
    _left_curverad = None
    _right_curverad = None
    if (line.detected):
        _left_fit, _right_fit, _left_fitx, _right_fitx, _ploty, _out_img = get_lane_lines_with_prior(warped, line.left_fit,
                                                                                               line.right_fit)
    # determine lane lines from scratch using sliding window
    else:
        _left_fit, _right_fit, _left_fitx, _right_fitx, _ploty, _out_img = get_lane_lines(warped)

    # determine lane curvature
    _left_curverad, _right_curverad = get_curvature(_ploty, _left_fit, _right_fit, _left_fitx, _right_fitx, line.xm_per_pix, line.ym_per_pix)

    # update lane detection
    line.checkDetected(_left_fit, _right_fit, _left_fitx, _right_fitx, _ploty, _left_curverad, _right_curverad)

    # determine vehicle position
    line.line_base_pos = get_vehicle_position(image, line.left_fitx, line.right_fitx, line.xm_per_pix)

    # draw lane findings
    result = draw(undist, image, warped, line.left_fitx, line.right_fitx, line.ploty, line.warp_minv, line.left_curverad, line.right_curverad, line.line_base_pos)

    #if(debug == 1):
        #if(not line.detected):
            #print(line.left_curverad, 'm', line.right_curverad, 'm')
            #cv2.imshow('result', result)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

    return result

if __name__ == "__main__":
    # Initialize line object
    line = Line()

    file_group = "test"
    test_path = os.path.join('test_images', file_group + '*.jpg')
    test_images = sorted(glob.glob(test_path))
    for idx, fname in enumerate(test_images):
        index = re.findall(r'\d+', fname)
        image = cv2.imread(fname)
        #line.detected = False
        #if (int(index[0]) > 1):
        #    line.detected = True
        result = AdvancedLaneLines(image, line, 1)
        #cv2.imshow('result', result)
        #cv2.waitKey(0)
    #cv2.destroyAllWindows()
