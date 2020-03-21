from ai.util import Storage
from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
import tensorflow as tf

"""
crim : crime rate, zn : residential area rate, indus : biz district rate, chas: close river 1 or 0
nox : nitrous oxide concentration, rm : a number of room per house, age : housing before 1940s rate,
dis : vocational employment center distance , rad: access to highway, tax, ptratio: teacher per students rate,
b : black man rate , lstat : lower layer ratio, medv : center value for housing price
"""
class Boston:
    boston : object
    def __init__(self):
        self.storage = Storage()

    @property
    def boston(self) -> object: return self._boston

    @boston.setter
    def boston(self, boston): self._boston = boston

    def initialize(self):
        this = self.storage
        (this.train_X, this.train_Y), (this.test_X, this.test_Y) = boston_housing.load_data()
        print(f'확률변수 X의 길이 : {len(this.train_X)}')
        print(f'확률변수 Y의 길이 : {len(this.train_Y)}')
        print(f'확률변수 X[0] : {this.train_X[0]}')
        print(f'확률변수 Y[0] : {this.train_Y[0]}')

    def standardization(self): # 데이터 전처리 .. 정규화
        this = self.storage
        x_mean = this.train_X.mean()
        x_std = this.train_X.std()
        this.train_X -= x_mean
        this.train_X /= x_std
        this.test_X -= x_mean
        this.test_X /= x_std
        y_mean = this.train_Y.mean()
        y_std = this.train_Y.std()
        this.train_Y -= y_mean
        this.train_Y /= y_std
        this.test_Y -= y_mean
        this.test_Y /= y_std

    def new_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(units=52, activation='relu', input_shape=(13, 1)),
            keras.layers.Dense(units=39, activation='relu'),
            keras.layers.Dense(units=26, activation='relu'),
            keras.layers.Dense(units=1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')

        return model

    def learn_model(self, model):
        this = self.storage
        history = model.fit(this.train_X, this.train_Y, epochs=25,
                            batch_size=32, validation_split=0.25,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,
                                                                        monitor='val_loss')])

        return history

    def eval_model(self, model):
        this = self.storage
        print(model.evaluate(this.test_X, this.test_Y))
        return this






