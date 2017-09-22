'''Load and structure the data from Udacity annotated set, KITTI and GTI.'''


# Third-party libs
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

# Own libs
import data_utils

# load Udacity crowd-ai dataset
IN_DIR = 'object-detection-crowdai/'
OUT_DIR = 'vehicles/vehicles/Udacity/'
OUT_DIR_NON = 'non-vehicles/non-vehicles/Udacity/'
GTI_IMG_SIZE = (64, 64)

def load_udacity_data():

    df_files = pd.read_csv(IN_DIR +'labels.csv', header=0)
    vehicles_df = df_files[(df_files['Label']=='Car') | (df_files['Label']=='Truck')].reset_index()
    vehicles_df = vehicles_df.drop('index', 1)
    vehicles_df['File_Path'] =  IN_DIR + vehicles_df['Frame']
    vehicles_df = vehicles_df.drop('Preview URL', 1)
    print(IN_DIR)
    print('BEFORE COLUMN REORDER: ','\n', vehicles_df.head())

    # re-order column names
    vehicles_df.columns = ['xmin','ymin','xmax','ymax', 'Frame','Label','File_Path']
    print('AFTER COLUMN REORDER:' ,'\n', vehicles_df.head())

    # augment, scale and save Udacity car/truck images in same scale as GTI, KITTI
    n_imgs = vehicles_df.shape[0]
    params = list()
    params.append(cv2.IMWRITE_PNG_COMPRESSION)
    params.append(2)
    for frame in range(n_imgs):
        f_name, img, bb_boxes = data_utils.extract_boxes(vehicles_df, frame,
            size=None, augmentation=False, trans_range=20, scale_range=20)
        #print(' Current box: ', bb_boxes)
        img_size = img.shape
        nb_boxes = bb_boxes.shape[0]
        for index, box in bb_boxes.iterrows():
            #print(' Current box: ', bb_boxes)
            x1 = np.max([box['xmin'], 0])
            y1 = np.max([box['ymin'], 0])
            x2 = np.min([box['xmax'], img_size[1]])
            y2 = np.min([box['ymax'], img_size[0]])
            #print('data_loader/ image size: ', img_size)
            #print('data_loader/ x1, y1...: ', x1, y1, x2, y2)
            extracted = img[y1:y2, x1:x2, :]
            # scale to same scale as GTI and KITTI
            #print('data_loader/extracted size', extracted.shape)
            try:
                extracted = cv2.resize(extracted, GTI_IMG_SIZE)
                extracted = cv2.cvtColor(extracted,cv2.COLOR_RGB2BGR)
                cv2.imwrite(OUT_DIR + f_name+str(index) +'.png', extracted, params)
            except cv2.error:
                print('filename:', f_name)
                print('extracted size', extracted.shape)
                print('data_loader/ image size: ', img_size)
                print('data_loader/ x1, y1...: ', x1, y1, x2, y2)
            #cv2.imwrite(OUT_DIR + f_name+str(index)+'.jpg' , img)
        # balance the Udacity dataset with non-vehicle images extracted from
        # same dataset
        # pick a box size:
        box_size_yx = (128, 128)
        # try only a certain number of times
        n_iter = 2 * nb_boxes
        while n_iter > 0:
            # pick a random box, focus on road part of image
            new_bb_y1 = np.random.random_integers(img_size[0]//3, img_size[0] - box_size_yx[0])
            new_bb_x1 = np.random.random_integers(0, img_size[1]- box_size_yx[1])
            new_bb_y2 = new_bb_y1 + box_size_yx[0]
            new_bb_x2 = new_bb_x1 + box_size_yx[1]
            new_box = ((new_bb_x1, new_bb_y1), (new_bb_x2, new_bb_y2))
            # determine the coordinates of the intersection rectangle with all
            # boxes
            intersections = []
            for index, box in bb_boxes.iterrows():
                x_left = max(box['xmin'], new_box[0][0])
                y_top = max(box['ymin'], new_box[0][1])
                x_right = min(box['xmax'], new_box[1][0])
                y_bottom = min(box['ymax'], new_box[1][1])
                # intersection area as percentage of new_box size
                intersection = ((x_right - x_left) * (y_bottom - y_top)) / (box_size_yx[0] * box_size_yx[1])
                #print('Points: ', x_left, x_right, y_bottom, y_top)
                #print(' INtersection: ', intersection)
                if x_right < x_left or y_bottom < y_top or intersection < 0.1:
                    intersections.append(False)
                else:
                    intersections.append(True)
            #print(intersections)
            if not any(intersections):
                #print("NO INTERSECTIONS")
                # no (meaningful) intersection with other boxes,
                # accept the new box
                extracted_new_box = img[new_bb_y1:new_bb_y2, new_bb_x1:new_bb_x2, :]
                #print('Extracted new box :', new_box)
                # scale to same scale as GTI and KITTI
                extracted_new_box = cv2.resize(extracted_new_box, GTI_IMG_SIZE)
                extracted_new_box = cv2.cvtColor(extracted_new_box,cv2.COLOR_RGB2BGR)
                #print('Extracted new box size:', extracted_new_box.shape)
                #print('INDEX:',index)
                cv2.imwrite(OUT_DIR_NON + f_name+str(n_iter) +'.png', extracted_new_box, params)
            n_iter -= 1


if __name__ == '__main__':
    load_udacity_data()




# augment GTI, KITTI images
