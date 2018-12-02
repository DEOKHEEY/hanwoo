
############# OPtimizer 추출 ###############

from __future__ import print_function
import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import math

import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler

############ Load data ##################

batch_size = 100
epochs = 50

categories = ["갈비","등심","부채살","살치","안심","양지","채끝"]
x = []
y = []
cn = 0
nb_class = len(categories)

for idx, cat in enumerate(categories):
    dir = "c:/data/Dataset/trainset/"+cat
    files = glob.glob(dir+"/*.jpg")
    label = [0 for i in range(nb_class)]
    label[idx] = 1
    
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB") 
        img = img.resize((128, 128))
        data = np.array(img)
        
        for angle in range(-15,15,5):
            img2 = img.rotate(angle)
            data = np.array(img2)
            x.append(data)
            y.append(label)
            cn =+1

            img3 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            data = np.array(img3)
            x.append(data)
            y.append(label)
            cn =+1

#rgb array 추출
x = np.array(x)
y = np.array(y)
#train&test 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y)
#scale 
x_train = x_train/256
x_test = x_test/256



###  CNN model 정의 ###

def cnn_model() :
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=x.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten()) 
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    return model 

### OPtimizer : SGD ,Adagrad, Adadelta, RMSprop, Adam ###

# CNN model에 SGD optimizer 적용

model1 = cnn_model()

model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=SGD(lr=0.001, decay=0.0),
              metrics=['accuracy'])
history1 = model1.fit(x_train, y_train, 
                     validation_data=(x_test, y_test), 
                     epochs=epochs, 
                     batch_size=batch_size,
                     verbose=2)


# CNN model에 Adagrad optimizer 적용

model2 = cnn_model()
model2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adagrad(lr=0.001, decay=0.0),
              metrics=['accuracy'])
history2 = model2.fit(x_train, y_train, 
                     validation_data=(x_test, y_test), 
                     epochs=epochs, 
                     batch_size=batch_size,
                     verbose=2)

# CNN model에 Adadelta optimizer 적용
model3 = cnn_model()
model3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adadelta(lr=0.001, decay=0.0),
              metrics=['accuracy'])
history3 = model3.fit(x_train, y_train, 
                     validation_data=(x_test, y_test), 
                     epochs=epochs, 
                     batch_size=batch_size)

# CNN model에 RMSprop optimizer 적용
model4 = cnn_model()
model4.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=RMSprop(lr=0.001, decay=0.0),
              metrics=['accuracy'])
history4 = model4.fit(x_train, y_train, 
                     validation_data=(x_test, y_test), 
                     epochs=epochs, 
                     batch_size=batch_size)
             

# CNN model에 Adam optimizer 적용
model5 = cnn_model()
model5.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(lr=0.001, decay=0.0),
              metrics=['accuracy'])
history5 = model5.fit(x_train, y_train, 
                     validation_data=(x_test, y_test), 
                     epochs=epochs, 
                     batch_size=batch_size) 
                    



### 모델 비교 ###

fig = plt.figure(figsize=(12,5))
plt.plot(range(epochs),history1.history['val_acc'],label='SGD')
plt.plot(range(epochs),history2.history['val_acc'],label='Adagrad')
plt.plot(range(epochs),history3.history['val_acc'],label='Adadelta')
plt.plot(range(epochs),history4.history['val_acc'],label='RMSprop')
plt.plot(range(epochs),history5.history['val_acc'],label='Adam')
plt.legend(loc=0)
plt.xlabel('epochs')
plt.xlim([0,epochs])
plt.ylabel('accuracy')
plt.grid(True)
plt.title("Comparing Model Accuracy")
plt.show()
fig.savefig('C:/data/Dataset/plot/compare-accuracy.jpg')
plt.close(fig)
