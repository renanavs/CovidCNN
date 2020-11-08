import os

import matplotlib.pyplot as plt
import keras
import pandas as pd
from PIL import Image
import tensorflow_addons as tfa
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adagrad
import numpy as np


class NeuralNetwork:

    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.y_test = []
        self.y_train = []
        self.x_evaluate_test = []
        self.y_evaluate_test = []
        self.learning_rate = 5e-3  # taxa de aprendizagem constante
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
                img = im.resize((200, 200))
                gray = img.convert('L')
                gray.save(path2 + '//' + file, 'png')

        for id_img in labels['Id_Imagem']:
            try:
                result = plt.imread(path2 + '//' + id_img)
                dataset_result.append(result)
            except Exception as e:
                print(e)

        self.x_train = np.array(dataset_result[0:110])
        self.x_test = np.array(dataset_result[110:150])
        self.x_evaluate_test = np.array(dataset_result[100:120])

        self.y_train = np.array(labels.saida[0:110])
        self.y_test = np.array(labels.saida[110:150])
        self.y_evaluate_test = np.array(labels.saida[100:120])

    def compile_model(self):
        # CNN
        model = Sequential()
        # 1 conv
        model.add(Conv2D(16, (3, 3), input_shape=(200, 200, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        # 2 conv
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        # 3 conv
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        # 4 conv
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        # 5 conv
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        # MLP
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer=Adagrad(lr=self.learning_rate), metrics=['accuracy'])
        model.summary()
        self.model = model

    def evaluate_model(self):
        model = self.model
        evaluate = model.evaluate(self.x_evaluate_test.reshape(-1, 200, 200, 1), self.y_evaluate_test.reshape(-1, 1))
        print(evaluate)

    def predict_model(self, id_img):
        imread = plt.imread('dataset_resized//' + id_img)
        model = self.model
        x_predict = np.array(imread).reshape(-1, 200, 200, 1)
        predict = model.predict(x_predict)
        print(predict)

    def train_model(self):
        model = self.model

        self.x_train = self.x_train.reshape(-1, 200, 200, 1)
        self.y_train = self.y_train.reshape(-1, 1)

        self.x_test = self.x_test.reshape(-1, 200, 200, 1)
        self.y_test = self.y_test.reshape(-1, 1)

        model.fit(
            self.x_train, self.y_train,
            batch_size=11,
            epochs=100,
            callbacks=[
                keras.callbacks.ModelCheckpoint('model_checkpoint', save_weights_only=True, save_freq=20),
                tfa.callbacks.TQDMProgressBar()
            ],
            validation_data=(self.x_test, self.y_test),
            shuffle=True,
            verbose=0,
            initial_epoch=0
        )
        # model.save('my_model.h5')


def execute_model(neural_net):
    neural_net
    neural_net.pre_process()  # carregar inputs
    neural_net.compile_model()
    neural_net.train_model()  # obs.: já invoca o compile model em seu body[
    print("evaluate model: ")
    neural_net.evaluate_model()
    print("predict test")
    neural_net.predict_model("01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg") # POSITIVO
    neural_net.predict_model("IM-0001-0001.jpeg")  # NEGATIVO
    neural_net.predict_model("NORMAL2-IM-0073-0001.jpeg")  # NEGATIVO
    neural_net.predict_model("IMG-COVID-00014.jpg")  # POSITIVO


nn = NeuralNetwork()
execute_model(nn)
