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
        self.left_curverad = None # meters
        self.right_curverad = None # meters

        #distance in meters of vehicle center from the line
        self.line_base_pos = None

    def check_lane_curvature(self, R, left=1):
        """
        Checks new radius of curvature `R` against the radius stored in the object.
        """
        R0 = self.right_curverad
        if(left == 1):
            R0 = self.left_curverad

        # Return true if there is no prior data
        if R0 is None:
            return True

        self.check = abs(R - R0) / R0

        return self.check <= 0.5  # Max change from frame to frame is 200%

    def checkDetected(self, _left_fit, _right_fit, _left_fitx, _right_fitx, _ploty, _left_curverad, _right_curverad):
        #difference in fit coefficients between last and new fits
        self.diffs_left = np.abs(self.left_fit-_left_fit)
        self.diffs_right = np.abs(self.right_fit- _right_fit)

        if(not self.saveNext):
            if(self.check_lane_curvature(_left_curverad, 1) and self.check_lane_curvature(_right_curverad, 0)):
            #if((self.diffs_left < 100).all() and (self.diffs_right < 100).all()):
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
                print("before: %.2f Now: %.2f" % (self.left_curverad, _left_curverad))
                print("before: %.2f Now: %.2f" %(self.right_curverad, _right_curverad))
                print("Not Detected Left!(%.2f,%.2f,%.2f)" % (self.diffs_left[0], self.diffs_left[1], self.diffs_left[2]))
                print("Not Detected Right!(%.2f,%.2f,%.2f)" % (self.diffs_right[0], self.diffs_right[1], self.diffs_right[2]))
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

def draw(undist, image, warped, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    return result

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
    #print(left_curverad, 'm', right_curverad, 'm')

    # update lane detection
    line.checkDetected(_left_fit, _right_fit, _left_fitx, _right_fitx, _ploty, _left_curverad, _right_curverad)

    # determine vehicle position
    vehicle_pos = image.shape[1] // 2
    middle = (line.left_fitx[-1] + line.right_fitx[-1]) // 2
    line.line_base_pos = (vehicle_pos - middle) * line.xm_per_pix

    # draw lane findings
    result = draw(undist, image, warped, line.left_fitx, line.right_fitx, line.ploty, line.warp_minv)

    # write curvature and position findings
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Radius of left line curvature: ' + str(line.left_curverad) + 'm', (50, 20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Radius of right line curvature: ' + str(line.right_curverad) + 'm', (50, 50), font, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(result, 'Vehicle position : %.2f m %s of center' % (abs(line.line_base_pos), 'left' if line.line_base_pos < 0 else 'right'), (50, 80),
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if(debug == 1):
        if(not line.detected):
            cv2.imshow('result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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
