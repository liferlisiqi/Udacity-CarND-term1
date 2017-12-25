# Data preprocessing

import csv
import cv2
import numpy as np
import os
import pickle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# Read data
lines = []
with open('./sample_data/data/driving_log.csv') as sample_file:
    sample_file.readline()
    reader = csv.reader(sample_file)
    for line in reader:
        lines.append(line)


# Get images/features and turning angles/labels
images = []
angles = []
for line in lines:
    center_path = './sample_data/data/IMG/' + line[0].split('/')[-1]
    center_img = cv2.imread(center_path)
    center_img = cv2.resize(center_img, None, fx=0.5, fy=0.5)
    center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)
    images.append(center_img)
    center_angle = float(line[3])
    angles.append(center_angle)
    
    left_path = './sample_data/data/IMG/' + line[1].split('/')[-1]
    left_img = cv2.imread(left_path)
    left_img = cv2.resize(left_img, None, fx=0.5, fy=0.5)
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    images.append(left_img)
    left_angle = float(line[3]) + 0.10
    angles.append(left_angle)
    
    right_path = './sample_data/data/IMG/' + line[2].split('/')[-1]
    right_img = cv2.imread(right_path)
    right_img = cv2.resize(right_img, None, fx=0.5, fy=0.5)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    images.append(right_img)
    right_angle = float(line[3]) - 0.10
    angles.append(right_angle)
    
X_train = np.array(images)
y_train = np.array(angles)
print(X_train.shape, y_train.shape)


# Save the preprocessing data
def save_data(pickle_file, start, end):
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump(
                    {
                        'X_train': X_train[start:end],
                        'y_train': y_train[start:end]
                    },
                    pfile, protocol=2)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
    print('Data cached in pickle file.')
    
save_data('./pre-data/1.pickle', 0, 4018)
save_data('./pre-data/2.pickle', 4018, 8036)
save_data('./pre-data/3.pickle', 8036, 12054)
save_data('./pre-data/4.pickle', 12054, 16072)
save_data('./pre-data/5.pickle', 16072, 20090)
save_data('./pre-data/6.pickle', 20090, 24108)





