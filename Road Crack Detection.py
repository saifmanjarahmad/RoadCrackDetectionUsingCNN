# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:33:09 2020

@author: Dell
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix

#Initialising the CNN
classifier = Sequential() #98.27

#step_1:Convolution
classifier.add(Convolution2D(32,(3,3),padding='same', input_shape=(64,64,3),activation='relu'))

#Step_2: Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a second convolutional layer
classifier.add(Convolution2D(32,(5,5),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a third convolutional layer
classifier.add(Convolution2D(64,(5,5),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a fourth convolutional layer
classifier.add(Convolution2D(128,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step_3: Flatten
classifier.add(Flatten())

#Step_3: Full_Connection
classifier.add(Dense(activation="relu", units=128))
classifier.add(Dense(activation="sigmoid", units=1))

#Step_4: Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Step_5: Fitting the CNN to the image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
  
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        r'D:\dataset1_1\training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
        )


validation_generator = test_datagen.flow_from_directory(
        r'D:\dataset1_1\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        shuffle= False
        )

#validation_generator = test_datagen.flow_from_directory(
        #r'D:\dataset\single_prediction',
        #target_size=(64, 64),
        #batch_size=32,
        #class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=19200,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=12800)

#Confution Matrix and Classification Report
Y_pred = classifier.predict_generator(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_pred=Y_pred>0.5
print('Confusion Matrix')
print(len(y_pred))
print(len(validation_generator.classes))
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['Cracked', 'NotCracked']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))




#Found 24000 images belonging to 2 classes.
#Found 8000 images belonging to 2 classes.
#24000/24000 [==============================] - 28496s 1s/step - loss: 0.0188 - accuracy: 0.9941 - val_loss: 0.0099 - val_accuracy: 0.9979
#Confusion Matrix
#8000
#8000
#[[3985   15]
 #[   2 3998]]
#Classification Report
 #             precision    recall  f1-score   support

  #   Cracked       1.00      1.00      1.00      4000
  #NotCracked       1.00      1.00      1.00      4000

   # accuracy                           1.00      8000
   #macro avg       1.00      1.00      1.00      8000
#weighted avg       1.00      1.00      1.00      8000
   
   
#Found 19200 images belonging to 2 classes.
#Found 12800 images belonging to 2 classes.
#19200/19200 [==============================] - 16007s 834ms/step - loss: 0.0221 - accuracy: 0.9929 - val_loss: 0.0100 - val_accuracy: 0.9972
#Confusion Matrix
#12800
#12800
#[[6378   22]
 #[  14 6386]]
#Classification Report
 #             precision    recall  f1-score   support

  #   Cracked       1.00      1.00      1.00      6400
  #NotCracked       1.00      1.00      1.00      6400

   # accuracy                           1.00     12800
   #macro avg       1.00      1.00      1.00     12800
#weighted avg       1.00      1.00      1.00     12800



















