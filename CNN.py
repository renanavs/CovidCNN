import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop, Adamax
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras import backend as K
from PIL import Image

NUM_CLASSES = 2
INIT_LR = 5e-2
dataset_result = []
expected_results = []
classes = ['Positivo', 'Negativo']
dataset_train = []
dataset_pred = []

print('COMEÇOOOU')
def pre_process():
    path1 = 'dataset/'
    path2 = 'dataset_resized/'
    listing = os.listdir(path1)
    inlist = os.listdir(path2)
    labels = pd.read_csv('labels.csv', sep=';')
    expected_results = labels.values

    for file in listing:
        im = Image.open(path1 + '//' + file)
        img = im.resize((180, 180))
        gray = img.convert('L')
        gray.save(path2 + '//' + file, 'png')

    for file in inlist:
        result = plt.imread(path2 + '//' + file)
        dataset_result.append(result)


def compile_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))  # the last layer with neuron for each class
    model.add(Activation("softmax"))  # output probabilities

    return model


# def TrainModel():
#  model = CompileModel()
#  model.summary()
#  model.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=INIT_LR),metrics=['accuracy'])
#  model.fit(
#    x_train2, y_train2,  # prepared data
#    batch_size=BATCH_SIZE,
#    epochs=EPOCHS,
#    callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler),
#               LrHistory(),
#               keras_utils.TqdmProgressCallback(),
#               keras_utils.ModelSaveCallback(model_filename)],
#    validation_data=(x_test2, y_test2),
#    shuffle=True,
#    verbose=0,
#    initial_epoch=last_finished_epoch or 0
# )


pre_process()

# TrainModel()
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
