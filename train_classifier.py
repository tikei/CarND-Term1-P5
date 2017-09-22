''' Train a classifier for image segmentation.'''
# Std libs
import glob
import time
import pickle
# import itertools

# Third-party libs
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Own libraries
import utils

# input directories
IN_DIR_1 = 'vehicles/vehicles/GTI_Far/'
IN_DIR_2 = 'vehicles/vehicles/GTI_Left/'
IN_DIR_3 = 'vehicles/vehicles/GTI_MiddleClose/'
IN_DIR_4 = 'vehicles/vehicles/GTI_Right/'
IN_DIR_5 = 'vehicles/vehicles/KITTI_extracted/'
IN_DIR_6 = 'vehicles/vehicles/Udacity/'
IN_DIRS = [IN_DIR_1, IN_DIR_2, IN_DIR_3, IN_DIR_4, IN_DIR_5, IN_DIR_6]

IN_DIR_NON_0 = 'hard_neg/hard_neg/cropped/'
IN_DIR_NON_1 = 'non-vehicles/non-vehicles/GTI/'
IN_DIR_NON_2 = 'non-vehicles/non-vehicles/Extras/'
IN_DIR_NON_3 = 'non-vehicles/non-vehicles/Udacity/'
IN_DIRS_NON = [IN_DIR_NON_1, IN_DIR_NON_2, IN_DIR_NON_3]

# model output

MODEL_OUT = 'trained_model.pkl'

# load data
cars = []
notcars = []

# set seed
rand_state = np.random.randint(0, 100)
np.random.seed(rand_state)

for directory in IN_DIRS:
    cars.append(glob.iglob(directory+'*.png'))
    # for image in images:
    #     cars.append(image)

for directory in IN_DIRS_NON:
    notcars.append(glob.iglob(directory+'*.png'))
    # for image in images:
    #     notcars.append(image)

hard_neg_imgs = glob.glob(IN_DIR_NON_0 + '*.png')

# chain the generators:
# cars = itertools.chain(*cars)
# notcars = itertools.chain(*notcars)

# limit number of images due to quadratic complexity of SVM
max_cars = 9000
max_notcars = 9000
# cars = itertools.islice(cars, max_cars)
# notcars = itertools.islice(notcars, max_notcars)

# data_info = utils.data_look(cars, notcars)

# print('Your function returned a count of',
#       data_info["n_cars"], ' cars and',
#       data_info["n_notcars"], ' non-cars')
# print('of size: ',data_info["image_shape"], ' and data type:',
#       data_info["data_type"])



# Plot the examples
# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(car_image)
# plt.title('Example Car Image')
# plt.subplot(122)
# plt.imshow(notcar_image)
# plt.title('Example Not-car Image')

# Parameter tuning
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svr = svm.SVC()
# clf = grid_search.GridSearchCV(svr, parameters)
# clf.fit(iris.data, iris.target)
params = {'color_space': 'LUV',
            'orient': 12,
            'pix_per_cell': 8,
            'cell_per_block': 2,
            'hog_channel': 'ALL',
            'spatial_size': (20,20),
            'hist_bins': 128,
            'spatial_feat': True,
            'hist_feat': True,
            'hog_feat': True}
# detection parameters:
# threshold = 3
# scales of 1. and 1.5
# The heatmap number of frames in a deque to average = 10.

# spatial = 32
# histbin = 128
# colorspace = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# orient = 9
# pix_per_cell = 8
# cell_per_block = 2
# hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
car_features = []
notcar_features = []
t = time.time()
cars_img_used = 0
notcars_img_used = 0
for gen in cars:
    for f_name in tqdm(gen, desc='Car images'):
        prob = np.random.uniform()
        if prob <= (1/3) and cars_img_used < max_cars:
            car_features.append(utils.extract_features(f_name, cspace=params['color_space'],
                orient=params['orient'], pix_per_cell=params['pix_per_cell'],
                cell_per_block=params['cell_per_block'],
                hog_channel=params['hog_channel'], hist_bins=params['hist_bins'],
                spatial_size=params['spatial_size']) )
            cars_img_used += 1
    if cars_img_used >= max_cars:
        break

for gen in notcars:
    for f_name in tqdm(gen, desc='Not car images'):
        # prob = np.random.uniform()
        if notcars_img_used < max_notcars:
            notcar_features.append(utils.extract_features(f_name, cspace=params['color_space'],
                orient=params['orient'], pix_per_cell=params['pix_per_cell'],
                cell_per_block=params['cell_per_block'],
                hog_channel=params['hog_channel'], hist_bins=params['hist_bins'],
                spatial_size=params['spatial_size']))
            notcars_img_used += 1
    if notcars_img_used >= max_notcars:
        break

X_hard_neg = []

for f_name in hard_neg_imgs:
    X_hard_neg.append(utils.extract_features(f_name,
                cspace=params['color_space'],
                orient=params['orient'], pix_per_cell=params['pix_per_cell'],
                cell_per_block=params['cell_per_block'],
                hog_channel=params['hog_channel'],
                hist_bins=params['hist_bins'],
                spatial_size=params['spatial_size']))


t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
print('Number of car images used: ', cars_img_used)
print('Number of NON-car images used: ', notcars_img_used+len(X_hard_neg))
print('Number of car features extracted: ', len(car_features))
print('Number of Not-car features extracted: ', len(notcar_features) + len(X_hard_neg))

X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# save scaler for classification on new data:
with open('x_scaler.pkl', 'wb') as out_f:
    pickle.dump(X_scaler, out_f, -1)



# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# add hard-negative mined data to the train data set:
scaled_X_hard_neg = X_scaler.transform(X_hard_neg)
X_train = np.vstack((X_train, scaled_X_hard_neg)).astype(np.float64)
y_hard_neg = np.zeros(len(X_hard_neg))
y_train = np.hstack((y_train, y_hard_neg))

# shuffle train data again after adding hard-negative mined examples
rand_state = np.random.randint(0, 100)
X_train, y_train = shuffle(X_train, y_train, random_state=rand_state)

print('Using:',params['orient'],'orientations',params['pix_per_cell'],
    'pixels per cell and', params['cell_per_block'],'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC

# LinearSVC(penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001, C=1.0, multi_class=’ovr’, fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
model_params = {'C': 1e-4}

svc = LinearSVC(C=model_params['C'])
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()

# save model
with open(MODEL_OUT, 'wb') as f_out:
    pickle.dump(svc, f_out, pickle.HIGHEST_PROTOCOL)

print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Training Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
