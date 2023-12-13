import cv2
import numpy as np
import matplotlib.pyplot as plt

# To deploy IPM based lane detection import the functions from this file and then feed the image 
# to the lane_detect_ipm_pipeline() function. 

def get_masked_image(img, boundaries = [150,0]):
    img_size = img.shape
    height_limit = boundaries[0]
    width_limit = boundaries[1]
    masked_img = img[height_limit:img_size[0],width_limit:img_size[1]]
    return masked_img


def get_ipm_image(masked_img):
    masked_img_shape = masked_img.shape
    M = np.array([[1.,6.17681159,0.],
                  [0.,3.63043478,0.],
                  [0.,0.02922705,1.]])
    ipm_img = cv2.warpPerspective(masked_img, M, (masked_img_shape[1], masked_img_shape[0]))
    return ipm_img

def get_saturation_img(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    hsv_img[hsv_img<80]=0 #threshold for saturation values
    return hsv_img[:,:,1]

def get_histogram_img(img):
    img_bin = np.zeros_like(img)
    img_bin[(img > 0)] = 1
    # a boundary of 60 is used to further reduce the ROI in the y direction to prevent errors due to misdetection of environment object
    histogram = np.sum(img_bin[60:,:], axis=0)  
    return histogram

def find_lane(img,mode=2,show=False):

    # Create a binary version of the warped image
    warped_bin = np.zeros_like(img)
    warped_bin[(img > 0)] = 1
    
    vis_img = img.copy()  # The image we will draw on to show the lane-finding process
    vis_img = cv2.cvtColor(vis_img,cv2.COLOR_GRAY2RGB)
    vis_img[vis_img > 0] = 255  # Max out non-black pixels so we can remove them later
    
    # Find the left an right right peaks of the histogram
    histogram = get_histogram_img(img)
    midpoint = histogram.shape[0]//2
    left_x = np.argmax(histogram[:midpoint])  # x-position for the left window
    right_x = np.argmax(histogram[midpoint:]) + midpoint  # x-position for the right window

    # Sliding Window Method  
    n_windows = 10
    win_height = warped_bin.shape[0]//n_windows
    margin = 25  # Determines how wide the window is 25
    pix_to_recenter = 50  # If we find this many pixels in our window we will recenter (too few would be a bad recenter)10

    # Find the non-zero x and y indices
    nonzero_ind = warped_bin.nonzero()
    nonzero_y_ind = np.array(nonzero_ind[0])
    nonzero_x_ind = np.array(nonzero_ind[1])

    left_line_ind, right_line_ind = [], []

    for win_i in range(n_windows):
        # Determine window corner points
        win_y_low = warped_bin.shape[0] - (win_i+1)*win_height
        win_y_high = warped_bin.shape[0] - (win_i)*win_height
        win_x_left_low = max(0, left_x - margin)
        win_x_left_high = left_x + margin
        win_x_right_low = right_x - margin
        win_x_right_high = min(warped_bin.shape[1]-1, right_x + margin)

        # Draw the windows on the vis_img
        rect_color, rect_thickness = (0, 255, 0), 3
        cv2.rectangle(vis_img, (win_x_left_low, win_y_high), (win_x_left_high, win_y_low), rect_color, rect_thickness)
        cv2.rectangle(vis_img, (win_x_right_low, win_y_high), (win_x_right_high, win_y_low), rect_color, rect_thickness)

        # Record the non-zero pixels within the windows
        left_ind = (
            (nonzero_y_ind >= win_y_low) &
            (nonzero_y_ind <= win_y_high) &
            (nonzero_x_ind >= win_x_left_low) &
            (nonzero_x_ind <= win_x_left_high)
        ).nonzero()[0]
        right_ind = (
            (nonzero_y_ind >= win_y_low) &
            (nonzero_y_ind <= win_y_high) &
            (nonzero_x_ind >= win_x_right_low) &
            (nonzero_x_ind <= win_x_right_high)
        ).nonzero()[0]
        left_line_ind.append(left_ind)
        right_line_ind.append(right_ind)

        # If there are enough pixels, re-align the window
        if len(left_ind) > pix_to_recenter:
            left_x = int(np.mean(nonzero_x_ind[left_ind]))
        if len(right_ind) > pix_to_recenter:
            right_x = int(np.mean(nonzero_x_ind[right_ind]))

    # Combine the arrays of line indices
    
    if((np.max(histogram[:midpoint])==0)):
        left_line_ind = []
    else:
        left_line_ind = np.concatenate(left_line_ind)

    if((np.max(histogram[midpoint:])==0)):
        right_line_ind = []
    else:    
        right_line_ind = np.concatenate(right_line_ind)

    # Gather the final line pixel positions
    left_x = nonzero_x_ind[left_line_ind]
    left_y = nonzero_y_ind[left_line_ind]
    right_x = nonzero_x_ind[right_line_ind]
    right_y = nonzero_y_ind[right_line_ind]     

    # Color the lines on the vis_img
    vis_img[left_y, left_x] = 254  # 254 so we can isolate the white 255 later
    vis_img[right_y, right_x] = 254  # 254 so we can isolate the white 255 later

    y_vals = np.linspace(0, warped_bin.shape[0]-1, warped_bin.shape[0])
    middle_y_vals = [int(a) for a in y_vals]

    if len(left_x) == 0:
        lflag=0
    else:
        lflag=1
        if mode == 2:
            left_fit = np.polyfit(left_y, left_x, 2)
            left_x_vals = left_fit[0]*y_vals**2 + left_fit[1]*y_vals + left_fit[2]
        else:
            left_fit = np.polyfit(left_y, left_x, 1)
            left_x_vals = left_fit[0]*y_vals + left_fit[1]
    if len(right_x)== 0:
        rflag=0
    else:
        rflag=1
        if mode == 2:
            right_fit = np.polyfit(right_y, right_x, 2)
            right_x_vals = right_fit[0]*y_vals**2 + right_fit[1]*y_vals + right_fit[2]
        else:
            right_fit = np.polyfit(right_y, right_x, 1)
            right_x_vals = right_fit[0]*y_vals + right_fit[1]

    if(rflag==1)and(lflag==1):
        if((right_x_vals[-1]-left_x_vals[-1])<=45):
            temp = (right_x_vals[-1]+left_x_vals[-1])*0.5
            if temp < 212:
                rflag=0
            else:
                lflag=0
    
    if (rflag == 0):
        # print('Detected Lanes - Left Found, Right Not Found')
        middle_x_vals = left_x_vals+60
        middle_x_vals = [int(a) for a in ((middle_x_vals))]
        left_x_vals = [int(a) for a in (left_x_vals)]
        vis_img[middle_y_vals, left_x_vals] = [250,0,0]
        vis_img[middle_y_vals, middle_x_vals] = [0,250,0]
    elif (lflag == 0):
        # print('Detected Lanes - Left Not Found, Right Found')
        middle_x_vals = right_x_vals-60
        middle_x_vals = [int(a) for a in ((middle_x_vals))]
        right_x_vals = [int(a) for a in (right_x_vals)]
        vis_img[middle_y_vals, right_x_vals] = [250,0,0]
        vis_img[middle_y_vals, middle_x_vals] = [0,250,0]
    else:
        # print('Detected Lanes - Left Found, Right Found')
        middle_x_vals = (left_x_vals+right_x_vals)/2
        middle_x_vals = [int(a) for a in ((middle_x_vals))]
        left_x_vals = [int(a) for a in (left_x_vals)]
        right_x_vals = [int(a) for a in (right_x_vals)]
        vis_img[middle_y_vals, left_x_vals] = [250,0,0]
        vis_img[middle_y_vals, right_x_vals] = [250,0,0]
        vis_img[middle_y_vals, middle_x_vals] = [0,250,0]

    if show:
        print("\nSliding Window Process Visualisation")
        plt.imshow(vis_img)
        plt.show()

    lane_lines_img = vis_img.copy()
    lane_lines_img[lane_lines_img >= 253] = 0  # This basically removes everything except the colored lane lines
    return lane_lines_img, middle_x_vals, middle_y_vals#, left_x_vals,right_x_vals

## Function to get lane angle
def get_lane_angle(x,y):
    y= np.flip(y)
    fit_cr = np.polyfit(x, y, 1)
    angle = np.rad2deg(np.arctan2(fit_cr[0],1)) 
    if angle<0:
        angle+=180
    return np.round(angle,3)

## Function to get lateral deviation
def get_deltax(x,lane_lines_img):
    pixeltometer=0.00326
    delta_x = (lane_lines_img.shape[1]/2)-x[-1]
    return np.round(delta_x*pixeltometer,4)

def get_curvature(x, y):
    pixeltometer=0.00326
    ym = np.array([yt*pixeltometer for yt in y])
    xm = np.array([xt*pixeltometer for xt in x])
    fit_cr = np.polyfit(ym, xm, 2)
    # Define y-value where we want radius of curvature - chose corresponding to the middle of the image
    y_eval = ym[45]
    # Calculation of R_curve (radius of curvature)
    curverad = ((1 + (2*fit_cr[0]*y_eval*pixeltometer + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad

def lane_detect_ipm_pipeline(img):
    # Takes a RGB image as input and returns the lane_lines_img for visualisation alongwith the predicted 
    # center lane coordinates, lane angle,lateral deviation and the radius of curvature. 

    masked_img = get_masked_image(img)
    warped_img = get_ipm_image(masked_img) 
    sat_img = get_saturation_img(warped_img)
    lane_lines_img, middle_x_vals, middle_y_vals = find_lane(sat_img,2,False)
    angle = get_lane_angle(middle_x_vals,middle_y_vals)
    lateral_deviation = get_deltax(middle_x_vals,lane_lines_img)
    curvature = get_curvature(middle_x_vals,middle_y_vals)
    lane_lines_img = cv2.addWeighted(lane_lines_img, 0.6, warped_img, 0.4, 0.0)
    lane_lines_img = cv2.putText(lane_lines_img,'Angle: '+str(angle), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255, 0), 1, cv2.LINE_AA)
    lane_lines_img = cv2.putText(lane_lines_img,'LD: '+str(lateral_deviation), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255, 0), 1, cv2.LINE_AA)
    lane_lines_img = cv2.putText(lane_lines_img,'R: '+str(curvature), (10, 80), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255, 0), 1, cv2.LINE_AA)


    return lane_lines_img,middle_x_vals,middle_y_vals,angle,lateral_deviation,curvature

