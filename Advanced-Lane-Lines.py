import os
import matplotlib.image as mpimg
from Helper_Functions import *

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
    test_images = glob.glob(test_path)
    if os.path.exists(undistort_path) and len([name for name in os.listdir(undistort_path)]) < 1:
        for idx, fname in enumerate(test_images):
            img = cv2.imread(fname)
            unimg = cal_undistort(img, mtx, dist)
            write_name = os.path.join(undistort_path,'undistorted_'+file_group+str(idx)+'.jpg')
            cv2.imwrite(write_name, unimg)
            print(write_name)

    # 3. Color/gradient threshold
    threshold_path = os.path.join('output_images', 'threshold')
    threshold_images = glob.glob(os.path.join(undistort_path, 'undistorted_'+file_group + '*.jpg'))
    if os.path.exists(threshold_path) and len([name for name in os.listdir(threshold_path)]) < 1:
        for idx, fname in enumerate(threshold_images):
            image = mpimg.imread(fname)
            result = pipeline(image)
            write_name = os.path.join(threshold_path, 'threshold_' + file_group + str(idx) + '.jpg')
            cv2.imwrite(write_name, result)
            print(write_name)

    # 4. Perspective transform

    # 5. Determine lane lines

    # 6. Determine lane curvature