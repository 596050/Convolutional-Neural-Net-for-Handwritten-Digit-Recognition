#Author: Pritish Yuvraj
import numpy
import matplotlib.pyplot as plt 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from scipy import misc
import numpy as np
from PIL import Image

#Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

#Reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

#Normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
	#Create Model
	model = Sequential()
	model.add(Convolution2D(32, 5, 5, border_mode = 'valid', input_shape = (1, 28, 28), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation = 'relu'))
	model.add(Dense(num_classes, activation = 'softmax'))
	#Compile Model
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return model

#build the model
model = baseline_model()
model.load_weights("./weights.hdf5")
#Fit the model
#checkpoint = ModelCheckpoint('weights.hdf5', monitor = 'val_acc', save_best_only = True, mode = 'max')
#callbacks_list = [checkpoint]
#model.fit(X_train, y_train, validation_data = (X_test, y_test), nb_epoch = 5, batch_size = 200, callbacks = callbacks_list)	
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

#Predict from user input
#no = int(raw_input("Enter a no: "))
#plt.subplot(221)
#plt.imshow(xTrain[no], cmap = plt.get_cmap('gray'))
#plt.show()
#a = model.predict_classes(X_train[no:no+1])
#for i in a:
#	print "The predicted No is ", i

x=Image.open('temp.jpg','r')
x=x.convert('L') #makes it greyscale
y=np.asarray(x.getdata(),dtype=np.float64).reshape((x.size[1],x.size[0]))

#<manipulate matrix y...>

y=np.asarray(y,dtype=np.uint8) #if values still in range 0-255! 
w=Image.fromarray(y,mode='L')
w.save('out.jpg')


arr = misc.imread('out.jpg')

X_test = arr.reshape(1, 1, 28, 28).astype('float32')	
temp = X_test.reshape(28, 28).astype('float32')
plt.subplot(221)
plt.imshow(temp, cmap = plt.get_cmap('gray'))
plt.show()
a = model.predict_classes(X_test)
for i in a:
	print "The predicted No is ", i
	print "Model is ", model.predict(X_test)
