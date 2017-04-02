import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from Helper_Functions import *
import cv2
import re
import numpy as np

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

if __name__ == '__main__':
    nx = 9
    ny = 6

    # 1. Camera Calibration
    cam_cal = 'cam_cal.npz'
    if not os.path.isfile(cam_cal):
        print('Calibrating camera...')
        mtx, dist = camera_setup()
        #img = cv2.imread('test_images/test1.jpg')
        #unimg = cal_undistort(img, mtx, dist)
        #write_name = 'output_images/test1.jpg'
        #cv2.imwrite(write_name, unimg)
        #cv2.imshow('img', unimg)
        #cv2.waitKey(500)
        #cv2.destroyAllWindows()
        np.savez(cam_cal, mtx=mtx, dist=dist)
    else:
        print('Loading calibration from %s...' %cam_cal)
        data = np.load(cam_cal)
        mtx = data['mtx']
        dist = data['dist']

    # 2. Distortion correction
    undistort_path = os.path.join('output_images','undistorted')
    file_group = 'test'
    test_path = os.path.join('test_images', file_group + '*.jpg')
    test_images = sorted(glob.glob(test_path))
    if os.path.exists(undistort_path) and len([name for name in os.listdir(undistort_path)]) < 1:
        for idx, fname in enumerate(test_images):
            index = re.findall(r'\d+', fname)
            img = cv2.imread(fname)
            unimg = cal_undistort(img, mtx, dist)
            write_name = os.path.join(undistort_path,'undistorted_'+file_group+str(index[0])+'.jpg')
            cv2.imwrite(write_name, unimg)
            print(write_name)

    # 3. Color/gradient threshold
    threshold_path = os.path.join('output_images', 'threshold')
    undistorted_images = sorted(glob.glob(os.path.join(undistort_path, 'undistorted_' + file_group + '*.jpg')))
    if os.path.exists(threshold_path) and len([name for name in os.listdir(threshold_path)]) < 1:
        for idx, fname in enumerate(undistorted_images):
            index = re.findall(r'\d+', fname)
            image = mpimg.imread(fname)
            color, result = pipeline(image)
            write_name = os.path.join(threshold_path, 'threshold_' + file_group + str(index[0]) + '.jpg')
            cv2.imwrite(write_name, result)
            #plt.imshow(result, cmap='gray', interpolation='bicubic')
            #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            #plt.savefig(write_name)
            #plt.show()
            #plt.close()

    # 4. Perspective transform
    perspective_threshold_path = os.path.join('output_images', 'perspective_threshold')
    threshold_images = sorted(glob.glob(os.path.join(threshold_path, 'threshold_'+file_group + '*.jpg')))
    warp_param = 'warp_param.npz'
    if os.path.exists(perspective_threshold_path) and len([name for name in os.listdir(perspective_threshold_path)]) < 1:
        for idx, fname in enumerate(threshold_images):
            index = re.findall(r'\d+', fname)
            image = mpimg.imread(fname)
            src, dst, warp_m, warp_minv = get_perspective_transform(image)
            np.savez(warp_param, src=src, dst=dst, warp_m=warp_m, warp_minv=warp_minv)
            write_name = os.path.join(perspective_threshold_path, 'perspective_threshold_' + file_group + str(index[0]) + '.jpg')
            result = cv2.warpPerspective(image, warp_m, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
            cv2.imwrite(write_name, result)
            if 0:
                plt.subplot(1, 2, 1)
                plt.hold(True)
                plt.imshow(image, cmap='gray')
                colors = ['r+', 'g+', 'b+', 'w+']
                for i in range(4):
                    plt.plot(src[i, 0], src[i, 1], colors[i])

                im2 = cv2.warpPerspective(image, warp_m, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
                plt.subplot(1, 2, 2)
                plt.hold(True)
                plt.imshow(im2, cmap='gray')
                for i in range(4):
                    plt.plot(dst[i, 0], dst[i, 1], colors[i])
                plt.savefig(write_name)
                plt.show()
                plt.close()
    else:
        print('Loading warp params from %s...' % warp_param)
        warp_data = np.load(warp_param)
        src = warp_data['src']
        dst = warp_data['dst']
        warp_m = warp_data['warp_m']
        warp_minv = warp_data['warp_minv']

    # 5. Determine lane lines
    perspective_threshold_images = sorted(glob.glob(os.path.join(perspective_threshold_path, 'perspective_threshold_' + file_group + '*.jpg')))
    if os.path.exists(perspective_threshold_path):
        for idx, fname in enumerate(perspective_threshold_images):
            index = re.findall(r'\d+', fname)
            img = mpimg.imread(fname)
            if 0:
                histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
                plt.plot(histogram)
                plt.show()
                plt.close()

            if 0:
                out_img = sliding_window_convolution(img)
                # Display the final results
                plt.imshow(out_img)
                plt.title('window fitting results')
                plt.show()
            else:
                if(int(index[0]) > 1) and False:
                    left_fit, right_fit, left_fitx, right_fitx, ploty, out_img = get_lane_lines_with_prior(img, left_fit, right_fit)
                else:
                    left_fit, right_fit, left_fitx, right_fitx, ploty, out_img = get_lane_lines(img)

                # 6. Determine lane curvature
                left_curverad, right_curverad = get_curvature(ploty, left_fit, right_fit, left_fitx, right_fitx)
                # Now our radius of curvature is in meters
                print(left_curverad, 'm', right_curverad, 'm')
                write_name = os.path.join('output_images',
                                                  'perspective_threshold_' + file_group + str(index[0]) + '.jpg')
                print(write_name)
                cv2.imwrite(write_name, out_img)
                plt.imshow(out_img)
                plt.plot(left_fitx, ploty, color='yellow')
                plt.plot(right_fitx, ploty, color='yellow')
                plt.xlim(0, 1280)
                plt.ylim(720, 0)
                plt.show()
                #plt.close()

