from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
from tensorflow import keras
from machine.random_number_maker import RandomNumberMaker
import numpy as np
import math
import matplotlib.pyplot as plt
class Boston:
    def __init__(self):
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = boston_housing.load_data()

    def show_dataset(self):
        print(f'확률변수 X 길이 : {len(self.train_X)}')
        print(f'확률변수 Y 길이 : {len(self.train_Y)}')
        print(f'확률변수 X[0] : {self.train_X[0]}')
        print(f'확률변수 Y[0] : {self.train_Y[0]}')

    def preprocessing(self): # 데이터전처리.. 정규화
        train_X = self.train_X
        train_Y = self.train_Y
        test_X = self.test_X
        test_Y = self.test_Y
        x_mean = train_X.mean()
        x_std = train_X.std()
        train_X -= x_mean
        train_X /= x_std
        test_X -= x_mean
        test_X /= x_std

        y_mean = train_Y.mean()
        y_std = train_Y.std()
        train_Y -= y_mean
        train_Y /= y_std
        test_Y -= y_mean
        test_Y /= y_std

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Dense(units=52, activation='relu', input_shape=(13,)),
            keras.layers.Dense(units=39, activation='relu'),
            keras.layers.Dense(units=26, activation='relu'),
            keras.layers.Dense(units=1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')
        # print(model.summary())
        x = np.arange(-5, 5, 0.01)
        sigmoid_x = [RandomNumberMaker.sigmoid(z) for z in x]
        tanh_x = [math.tanh(z) for z in x]
        relu = [ 0 if z < 0 else z for z in x]
        plt.axhline(0, color = 'gray')
        plt.axvline(0, color = 'gray')
        plt.plot(x, sigmoid_x, 'b-', label='sigmoid')
        plt.plot(x, tanh_x, 'r--', label='tanh')
        plt.plot(x, relu, 'g.', label='relu')
        plt.legend()
        plt.show()



