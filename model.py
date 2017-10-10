import csv
import cv2
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from PIL import Image
from random import shuffle
from keras import backend
from keras.models import Model
import matplotlib.pyplot as plt

backend.set_image_dim_ordering ('tf')

BATCH_SIZE = 32

samples = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Just for Reference
print("Total Sample Lenth = ", len(samples))
print("Train Sample Length =", len(train_samples))
print("Validation Sample Length = ", len(validation_samples))

def display_image(image):#Function to Display Image
	cv2.imshow('download',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def process_image(image):#Image Pre-processing
	image = image[65:140,:,:] #crop
	image = cv2.GaussianBlur(image, (3,3), 0)
	image = cv2.resize(image, (200, 66), cv2.INTER_AREA)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
	return image

def generator(samples_in, batch_size = BATCH_SIZE):
	shuffle(samples_in)
	correction_angle = 0.25
	correction = [0, correction_angle, -correction_angle]
	num_samples = len(samples_in)
	while True: # Loop forever so the generator never terminates
		for offset in range(0, num_samples, batch_size):
			
			batch_samples = samples_in[offset:offset+batch_size]
			images = []
			angles = []
			for i in range(3):
				for batch_sample in batch_samples:
					steering_angle = float(batch_sample[3])
					
					angle = steering_angle + correction[i]
					cname = './data/IMG/'+batch_sample[i].split('\\')[-1] #Use \\ for new data
					image = process_image(cv2.imread(cname))
					
					images.extend([image, np.fliplr(image)])
					angles.extend([angle, -angle])
					
			X_train = np.array(images)
			y_train = np.array(angles)
			yield X_train, y_train

train_generator = generator(train_samples, batch_size = BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size = BATCH_SIZE)

#input image size
ch, row, col = 3, 66, 200

def behavior_model():#Model Design
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(row, col, ch),output_shape=(row, col, ch)))
	model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
	model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
	model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
	model.add(Conv2D(64, 3, 3, activation='relu'))
	model.add(Conv2D(64, 3, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1164, activation='relu'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1))
	model.summary()
	return model


def train_model(model):#Train the model

	model.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
	history_object = model.fit_generator(train_generator,
						samples_per_epoch = len(train_samples),
						nb_epoch = 5,
						validation_data=validation_generator,
						nb_val_samples=len(validation_samples),
						verbose=1)
	
	#model.save('model.h5') #Commented so that the existing model is not erased
	
	print(history_object.history.keys())### print the keys contained in the history object
	
	### plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()
	


def main():#Load the model and train the model
	model = behavior_model()
	train_model(model)

if __name__ == '__main__':
	main()

