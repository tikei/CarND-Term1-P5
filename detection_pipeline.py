''' A pipeline to detect cars in images/video.'''

# Std libs
import glob
import time
import pickle
import itertools
from collections import deque

# Third-party libs
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
from moviepy.editor import VideoFileClip

# Own libraries
import utils
VIDEO_IN ='project_video.mp4'
# VIDEO_IN = 'project_video.mp4'
VIDEO_OUT = 'output_images/project_video_output.mp4'

# load the scaler from training data:
with open('x_scaler.pkl', 'rb') as inp_f:
    x_scaler = pickle.load(inp_f)
# check scaler
print('Scaler loaded: ', type(x_scaler) )

# load the pre-trained model:
with open('trained_model_v5_2.pkl', 'rb') as inp_f:
    trained_model = pickle.load(inp_f)
# check model
print('Model loaded: ', type(trained_model) )



# image, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, col_space='RBG2LUV', cells_per_step=2, win_size=(64,64), scale_factor=1.5


func_params = {'ystart': 400, 'ystop': 650,
            'svc':trained_model, 'X_scaler': x_scaler,
            'col_space':'RGB2LUV',
            'orient':12,
            'pix_per_cell':8,
            'cell_per_block':2,
            'spatial_size': (20,20),
            'hist_bins':128,
            'scale_factor': [],
            'cells_per_step': 2}
            # 'hog_channel':'ALL',
            # 'spatial_feat':True,
            # 'hist_feat':True,
            # 'hog_feat':True}

in_file = 'test_images/test1.jpg'
inp_img = cv2.imread(in_file)
inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)


# scale_list = [1, 1.1, 1.2, 1.5]# , 2.]
scale_list = [1, 1.25, 1.5, 1.75, 2, 2.5]
func_params['scale_factor'] = scale_list
func_params['decision_threshold'] = 0.2

### Pipeline test:
# t1 = time.time()
# print('Function parameters: ', func_params)

### PIPELINE TESTING with single images
# boxes = utils.find_cars(inp_img, **func_params)
# inp_image = np.copy(inp_img)
# out_img = utils.draw_labeled_bboxes(inp_image, boxes)
# Visualize heatmap
# heat = np.zeros_like(inp_image[:,:,0]).astype(np.int16)
# heat = utils.add_heat(heat, boxes)
# heatmap = np.clip(heat, 0, 255)
# # Visualize labels:
# labels = utils.label(heatmap)
# outp_bboxes = utils.get_boxes(labels)
# out_img = utils.draw_labeled_bboxes(inp_image, outp_bboxes)
# utils.display_img(inp_img, out_img) #heatmap, gray=True) #out_img)

# t2 = time.time()
# print(round(t2-t1, 2), 'Seconds to extract find cars in img: ', in_file)


### VIDEO STREAM
# average heatmap over n frames
clip1 = VideoFileClip(VIDEO_IN)# .subclip(19,25)
# out_clip = clip1.fl_image(lambda x : utils.detection_pipe(x,
#             heat_deq, box_deq, car_list, func_params))
heat_deq = deque(maxlen=12)
heat_thresh = 3
car_detect_thresh = 6
bbox_avg = 10
img_processor = utils.VideoProcessor(func_params, heat_deq ,heat_thresh,
                                    car_detect_thresh, bbox_avg)

out_clip = clip1.fl_image(img_processor.detection_pipe)

tqdm(out_clip.write_videofile(VIDEO_OUT, audio=False))
