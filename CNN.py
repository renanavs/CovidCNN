import os

import matplotlib.pyplot as plt
import keras
import pandas as pd
from PIL import Image
import tensorflow_addons as tfa
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization, AveragePooling2D, GlobalAveragePooling2D


class NeuralNetwork:

    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.y_test = []
        self.y_train = []
        self.x_evaluate_test = []
        self.y_evaluate_test = []
        self.NUM_CLASSES = 2  # POSITIVO, NEGATIVO
        self.learning_rate = 1e-5  # taxa de aprendizagem constante
        self.dataset_result = []
        self.expected_results = []
        self.classes = ['Positivo', 'Negativo']
        self.model = None

    def pre_process(self):
        path1 = 'dataset/'
        path2 = 'dataset_resized/'
        listing = os.listdir(path1)
        if not os.path.isdir(path2):
            os.mkdir(path2)
            inlist = os.listdir(path2)
        else:
            inlist = os.listdir(path2)
        dataset_result = []
        labels = pd.read_csv('labels.csv', sep=';')
        # expected_results = labels.values

        if len(inlist) == 0:
            for file in listing:
                im = Image.open(path1 + '//' + file)
                img = im.resize((256, 256))
                gray = img.convert('L')
                gray.save(path2 + '//' + file, 'png')

        for id_img in labels['Id_Imagem']:
            try:
                result = plt.imread(path2 + '//' + id_img)
            except Exception as e:
                print(e)
            dataset_result.append(result)

        self.x_train = np.array(dataset_result[0:110])
        self.x_test = np.array(dataset_result[110:150])
        self.x_evaluate_test = np.array(dataset_result[150:160])

        self.y_train = np.array(labels.saida[0:110])
        self.y_test = np.array(labels.saida[110:150])
        self.y_evaluate_test = np.array(labels.saida[150:160])

    def compile_model(self):
        # CNN
        model = Sequential()
        # model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", input_shape=input_shape))
        # model.add(Activation('relu'))
        # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        # model.add(Activation('relu'))
        # # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        # # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # # MLP
        # model.add(Flatten())
        # model.add(Dense(256))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(1))
        # model.add(Activation("sigmoid"))
        # model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        #         # model.summary()
        model.add(Conv2D(32, (5, 5), strides=(1, 1), input_shape=(256, 256, 1)))
        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(AveragePooling2D((3, 3)))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(300, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        model.summary()
        self.model = model

    def evaluate_model(self):
        model = self.model
        predict_value = model.predict(self.x_evaluate_test.reshape(-1, 256, 256, 1))
        print(predict_value)
        print(self.y_evaluate_test)

    def train_model(self):
        model = self.model

        self.x_train = self.x_train.reshape(-1, 256, 256, 1)
        self.y_train = self.y_train.reshape(-1, 1)

        self.x_test = self.x_test.reshape(-1, 256, 256, 1)
        self.y_test = self.y_test.reshape(-1, 1)

        model.fit(
            self.x_train, self.y_train,  # prepared data
            batch_size=6,
            epochs=10,
            callbacks=[
                keras.callbacks.ModelCheckpoint('model_checkpoint', save_weights_only=True, save_freq=20),
                tfa.callbacks.TQDMProgressBar()
            ],
            validation_data=(self.x_test, self.y_test),
            shuffle=True,
            verbose=0,
            initial_epoch=0
        )
        model.save('my_model.h5')


def execute_model():
    neural_net = NeuralNetwork()
    neural_net.pre_process()  # carregar inputs
    neural_net.compile_model()
    neural_net.train_model()  # obs.: já invoca o compile model em seu body
    neural_net.evaluate_model()


if __name__ == "__main__":
    execute_model()
