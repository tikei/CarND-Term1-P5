''' Utility functions for detecting cars in images/video.
    Augmentation functions courtesy to Vivek Yadav.

    Feature extraction functions (HOG, spatial bining, color histograms)
    taken from Udacity course materials.
'''

import time
from collections import deque
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
# from skimage.transform import pyramid_gaussian
from scipy.ndimage.measurements import label

# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    test_img = mpimg.imread(car_list[1])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = test_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = test_img.dtype
    # Return data_dict
    return data_dict

def display_img(original, modified, save=False, out_file=None, colmap=None, img_headings=[]):
    if img_headings == []:
        img_headings = ['Original Image', 'Modified']

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(original)
    ax1.set_title(img_headings[0], fontsize=40)
    if colmap:
        ax2.imshow(modified, cmap=colmap)
    else:
        ax2.imshow(modified)
    ax2.set_title(img_headings[1], fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

    if save:
        # Save example
        _ = cv2.imwrite(out_file, modified)

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def color_hist(img, nbins=32, bins_range=(0, 256), return_val='feat_only'):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], nbins, bins_range)
    ghist = np.histogram(img[:,:,1], nbins, bins_range)
    bhist = np.histogram(img[:,:,2], nbins, bins_range)
    # Generating bin centers
    bin_centers = (rhist[1][1:] + rhist[1][0:len(rhist[1])-1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    if return_val == 'feat_only':
        return hist_features
    else:
        return rhist, ghist, bhist, bin_centers, hist_features

def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                    vis=False, feature_vec=True):
    ''' Extract histogram of oriented gradients (HOG) features from image.'''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                            pixels_per_cell=(pix_per_cell, pix_per_cell),
                            cells_per_block=(cell_per_block, cell_per_block),
                            transform_sqrt=True, visualise=vis,
                            feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=
                       (pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True, visualise=vis,
                       feature_vector=feature_vec)
        return features


def extract_features(file_name, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2,
                     hog_channel=0, spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256),
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    # features = []
    # Iterate through the list of images
    #for file_name in imgs:
        # Read in each one by one
    file_features = []
    ### Check range of values when reading png with mpimg !!!
    #image = mpimg.imread(file_name)
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    else: feature_image = np.copy(image)
    if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)

    if spatial_feat == True:
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        file_features.append(hist_features)
    # Append the new feature vector to the features list
    # features.append(np.concatenate(file_features))

    # Return list of feature vectors

    return np.concatenate(file_features)



def find_cars(img, ystart, ystop, svc, X_scaler, orient, pix_per_cell,
              cell_per_block, spatial_size, hist_bins,
              col_space='RBG2LUV', cells_per_step=2,
              win_size=(64,64), scale_factor=[None],
              decision_threshold=0.5):
    ''' A single function that can extract features using hog
    sub-sampling and make predictions.'''

    # draw_img = np.copy(img) # for img with drawn boxes output
    bbox_list = []
    # img = img.astype(np.float32)/255 # scaling when image read with mpimg.
    # print('find_cars/ystart, ystop: ', ystart, ystop)

    # img_tosearch = img[ystart:ystop,:,:]
    # ctrans_tosearch = convert_color(img_tosearch, conv=col_space)
    # y_adjust_min = 40
    y_adjust_max = 150
    # n_scales = len(scale_factor)
    # adj_y_min = int(y_adjust_min / n_scales)
    # adj_y_max = int(y_adjust_max / n_scales)

    for scale in scale_factor:
        # crop image further to adjust for
        # scale/ size of cars in areas of img
        # focusing on areas of image at relevant scale where the cars are
        # likely to appear

        # new_ystart =  np.max([ystart + y_adjust_min, ystart] )
        y_adjust_max = int(y_adjust_max // (scale ** 2))
        new_ystop = ystop - y_adjust_max
        img_tosearch = img[ystart : new_ystop,:,:]

        # y_adjust_min -= adj_y_min
        # y_adjust_max = int(y_adjust_max // (scale **2)) #adj_y_max

        ctrans_tosearch = convert_color(img_tosearch, conv=col_space)

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch,
                (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = win_size[0]
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        # cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                xpos = xb*cells_per_step
                ypos = yb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((hog_features, spatial_features, hist_features))) # .reshape(1, -1))
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                # test_prediction = svc.predict(test_features)
                test_prediction = svc.decision_function(test_features)

                if test_prediction > decision_threshold:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                    bbox_list.append(((xbox_left, ytop_draw+ystart),
                        (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                    # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

    return bbox_list # draw_img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def get_boxes(labels):
    ''' Get bounding boxes from labels.'''
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    return bboxes



def draw_labeled_bboxes(img, bboxes):
    ''' Helper function to draw bounding boxes on image.'''
    for bbox in bboxes:
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 3)
    # Return the image
    return img



class VideoProcessor(object):
    def __init__(self, img_processing_parameters, heatmap_deq, heatmap_thresh,
                car_track_thresh, bbox_lowpass_filter):
        self.func_params = img_processing_parameters
        self.heat_deq = heatmap_deq
        self.heat_thresh = heatmap_thresh
        self.car_detect_threshold = car_track_thresh
        self.bbox_avg = bbox_lowpass_filter
        self.car_list = []

    def detection_pipe(self, img): # heatmap_deq, box_deq, self.car_list, detect_params):
        '''Detection pipeline to draw boxes around vehicles.'''

        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        box_list = find_cars(img, **self.func_params) #detect_params)

        ### FOR DEBUGGING
        # # save boxes for debugging
        # OUT_DIR = 'hard_neg/'
        # import os
        # i = 0
        # while os.path.exists(os.path.join(OUT_DIR, ("detected_%s.png" % i))):
        #     i += 1
        # for box in box_list:
        #     y1 = box[0][1]
        #     y2 = box[1][1]
        #     x1 = box[0][0]
        #     x2 = box[1][0]
        #     extracted = img[y1:y2, x1:x2, :]
        #     extracted = cv2.cvtColor(extracted,cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(OUT_DIR + 'detected_' + str(i) + '.png', extracted)
        #     i += 1

        heat = add_heat(heat, box_list)

        # add to identified heatmaps from previous frames
        self.heat_deq.append(heat)

        # integrate the heatmaps over the previous n frames:
        integrated_heat = np.sum(self.heat_deq, axis=0)
        heat = apply_threshold(integrated_heat, self.heat_thresh) #4)

        # Visualize images and bounding boxes
        # draw_img = draw_labeled_bboxes(np.copy(img), box_list)
        # display_img(img, draw_img)
        # display_img(img, heat, gray=True)

        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        bboxes = get_boxes(labels)

        bboxes_to_draw = [] # bboxes
        box_map = {k:False for k in bboxes}

        # avg_boxes = np.mean(box_deq, axis=0).astype(np.int16)
        iou_threshold = 0.5
        if self.car_list == []:
            # no cars tracked yet, start new tracking:
            for box in bboxes:
                lab = hash(box) # unique ID, initial box coordinates
                car = Car(lab, self.car_detect_threshold, self.bbox_avg)# 5, 5)
                car.set_values(box)
                self.car_list.append(car)
                bboxes_to_draw.append(car.draw_box)
                box_map[box] = True
        else:
            # check if box belongs to an already detected car:
            for box in bboxes:
                for car in self.car_list:
                    iou_value = IoU(car.draw_box, box)
                    # print('/detection_pipe/ IoU value: ', iou_value)
                    if iou_value > iou_threshold:
                        # print('existing car')
                        # the box belongs to that car
                        car.set_values(box)
                        bboxes_to_draw.append(car.draw_box)
                        box_map[box] = True
            # box map to track used boxes:
            for box, detected in box_map.items():
                if not detected:
                    # new car:
                    # print('new car')
                    lab = hash(box) # unique ID, initial box coordinates
                    new_car = Car(lab, self.car_detect_threshold,
                                  self.bbox_avg) #5, 5)
                    new_car.set_values(box)
                    self.car_list.append(new_car)
                    bboxes_to_draw.append(new_car.draw_box)

        self.car_list = [car for car in self.car_list if car.check_tracking()]

        if len(bboxes_to_draw) > 0: #len(self.heat_deq) > 0:
            draw_img = draw_labeled_bboxes(np.copy(img), bboxes_to_draw)
             #avg_boxes)# bboxes)#
            return draw_img
        else:
            return np.copy(img)



class Car(object):
    ''' A class to track and follow car information.'''
    def __init__(self, car_label, det_req, avg_len):
        self.car_id = car_label
        self.detected = False
        self.detections = det_req
        # self.bounding_boxes = deque(maxlen=det_req)
        self.all_boxes = deque(maxlen=avg_len)
        self.avg_box = 0
        self.draw_box = None
        self.required_detections = det_req

    def set_values(self, bbox):
        self.detections += 2
        # self.bounding_boxes.append(bbox)
        self.all_boxes.append(bbox)
        self.avg_box = np.mean(self.all_boxes, axis=0).astype(np.int16)
        bbx = self.avg_box
        self.draw_box = ((bbx[0][0], bbx[0][1]), (bbx[1][0], bbx[1][1]))

    def check_tracking(self):
        self.detections -= 1
        if self.detections >= self.required_detections:
            self.detected = True
            # print('check_tracking/ detected: ', self.detected)
        if self.detections <= 0:
                # stop tracking
                self.detected = False
        return self.detected

    # def get_draw_box(self, bbox):
    #     self.avg_box = np.mean(self.all_boxes, axis=0).astype(np.int16)
    #     bbx = self.avg_box

    #     self.draw_box = ((bbx[0][0], bbx[0][1]), (bbx[1][0], bbx[1][1]))
    #     return self.draw_box


def IoU(boxA, boxB):
    ''' Computes intersection over union to determine overlap between
    two bounding boxes on an image.'''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
    boxBArea = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou




#################### Unused code: ######################







#######################################
# From PyImageSearch:

def pyramid(image, scale=1.5, minSize=(128, 128)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        imshape = image.shape
        image = cv2.resize(image, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def car_finder(image, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, col_space='RBG2LUV', cells_per_step=2, win_size=(64,64), scale_factor=1.5):

    t1 = time.time() # t1

    draw_img = np.copy(image)

    (winW, winH) = win_size

    img_tosearch = image[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv=col_space)
    # print('car_finder/img: ', ctrans_tosearch[0,0,:])
    # nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    t2 = time.time() # t2
    print(round(t2-t1, 2), 'Seconds in first section ')
    # loop over the image pyramid
    new_y_scale = 220
    for resized in pyramid(ctrans_tosearch, scale=scale_factor):
        t1 = time.time()

        # further resize to adjust for different scale
        new_y_scale = int(new_y_scale // (scale_factor**2))
        y_max = resized.shape[0]
        # print('y_max ', y_max )
        # print('new_y_scale: ', new_y_scale)
        # resized = resized[0: (y_max-new_y_scale),:,:]
        ch1 = resized[:, :, 0]
        ch2 = resized[:, :, 1]
        ch3 = resized[:, :, 2]

        # calclate HOG for resized image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        t2 = time.time() # t2
        print(round(t2-t1, 2), 'Seconds extracting HOG ')
        # loop over the sliding window for each layer of the pyramid
        nblocks_per_window = (winH // pix_per_cell) - cell_per_block + 1
        # print('Image size: ', resized.shape)
        t1 = time.time()
        for (xpos, ypos, window) in sliding_window(resized, stepSize=cells_per_step, windowSize=(winW, winH)):

            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # translate xpos, ypos into HOG blocks coordinates

            xp = (xpos // winW) * cells_per_step
            yp = (ypos // winH) * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[yp:yp+nblocks_per_window , xp:xp+ nblocks_per_window ].ravel()
            hog_feat2 = hog2[yp:yp+nblocks_per_window , xp:xp+ nblocks_per_window].ravel()
            hog_feat3 = hog3[yp:yp+nblocks_per_window , xp:xp+ nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            # Extract the image patch
            subimg = window

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            # print('Feature dimensions: ', hog_features.shape, spatial_features.shape, hist_features.shape)

            # Scale features and make a prediction
            try:
                test_features = X_scaler.transform(np.hstack((hog_features, spatial_features, hist_features)))
            except ValueError:
                raise ValueError
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            # test_prediction = svc.predict(test_features)
            test_prediction = svc.decision_function(test_features)
            # if test_prediction > 0.5:
            #     print('car/finder/decision function: ', test_prediction)


            # print(' car_finder/Test prediction: ',test_prediction)

            if test_prediction > 0.7:
                # cv2.rectangle(draw_img, (xpos, ypos), (xpos + winW, ypos + winH), (0, 0, 255), 6)

                xleft = xp * pix_per_cell
                ytop = yp * pix_per_cell
                xbox_left = np.int(xleft * scale_factor)
                ytop_draw = np.int(ytop * scale_factor)
                win_draw = np.int(winW * scale_factor)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                    (xbox_left + win_draw, ytop_draw + win_draw + ystart),
                    (0,0,255),6)
        t2 = time.time()
        print(round(t2-t1, 2), 'Seconds per window of size ', resized.shape)
    return draw_img





