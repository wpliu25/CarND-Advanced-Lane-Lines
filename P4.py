#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from Helper_Functions import *
import re

# Calibrations
cam_cal = 'cam_cal.npz'
warp_param = 'warp_param.npz'
if not os.path.isfile(cam_cal):
    print('Calibrating camera...')
    mtx, dist = camera_setup()
    np.savez(cam_cal, mtx=mtx, dist=dist)
else:
    print('Loading calibration from %s...' % cam_cal)
    data = np.load(cam_cal)
    mtx = data['mtx']
    dist = data['dist']
    print('Loading warp params from %s...' % warp_param)
    warp_data = np.load(warp_param)
    src = warp_data['src']
    dst = warp_data['dst']
    warp_m = warp_data['warp_m']
    warp_minv = warp_data['warp_minv']

left_fit = None
right_fit = None
left_fitx = None
right_fitx = None
ploty = None

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meteres per pixel in x dimension

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

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

def AdvancedLaneLines(image, use_prior=0, debug=0):
    global left_fit
    global right_fit
    global left_fitx
    global right_fitx
    global ploty
    global warp_minv

    undist = cal_undistort(image, mtx, dist)
    color, threshold = pipeline(undist)
    warped = cv2.warpPerspective(threshold, warp_m, (threshold.shape[1], threshold.shape[0]), flags=cv2.INTER_LINEAR)

    if (use_prior == 1):
        left_fit, right_fit, left_fitx, right_fitx, ploty, out_img = get_lane_lines_with_prior(warped, left_fit,
                                                                                                      right_fit)
    else:
        left_fit, right_fit, left_fitx, right_fitx, ploty, out_img = get_lane_lines(warped)

    # Determine lane curvature
    left_curverad, right_curverad = get_curvature(ploty, left_fit, right_fit, left_fitx, right_fitx)
    #print(left_curverad, 'm', right_curverad, 'm')

    veh_pos = image.shape[1] // 2
    middle = (left_fitx[-1] + right_fitx[-1]) // 2
    dx = (veh_pos - middle) * xm_per_pix 

    result = draw(undist, image, warped, left_fitx, right_fitx, ploty, warp_minv)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Radius of left line curvature: ' + str(left_curverad) + 'm', (50, 20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Radius of right line curvature: ' + str(right_curverad) + 'm', (50, 50), font, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(result, 'Vehicle position : %.2f m %s of center' % (abs(dx), 'left' if dx < 0 else 'right'), (50, 80),
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result

if __name__ == "__main__":
    file_group = "test"
    test_path = os.path.join('test_images', file_group + '*.jpg')
    test_images = sorted(glob.glob(test_path))
    for idx, fname in enumerate(test_images):
        index = re.findall(r'\d+', fname)
        image = cv2.imread(fname)
        use_prior = 0
        if (int(index[0]) > 1):
            use_prior = 1
        result = AdvancedLaneLines(image, use_prior)
        cv2.imshow('result', result)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
