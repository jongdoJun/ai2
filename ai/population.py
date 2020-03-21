from ai.util import Storage
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

class Population:
    population_inc : []
    population_old : []

    @property
    def population_inc(self) -> object: return self._population_inc
    @population_inc.setter
    def population_inc(self, population_inc): self._population_inc = population_inc
    @property
    def population_old(self) -> object: return self._population_old
    @population_inc.setter
    def population_old(self, population_old): self._population_old = population_old

    def __init__(self):
        self.storage = Storage()

    def initialize(self):
        # 4.1 지역별 인구증가율과 고령인구비율 시각화
        self.population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27,
        0.02, -0.76, 2.66]
        self.population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94,
        12.83, 15.51, 17.14, 14.42]

    def population_without_outlier(self):#15.17 을 제거함
        self.population_inc = self.population_inc[:5] + self.population_inc[6:]
        self.population_old = self.population_inc[:5] + self.population_inc[6:]

    def population_with_regression_(self):#퍼센트런
        #최소제곱법으로 회귀선 구하기 대문자 X,Y 는 확률변수
        X = self.population_inc
        Y = self.population_old
        x_bar = sum(X) / len(X) # 평균
        y_bar = sum(Y) / len(Y)
        a = sum([( y - y_bar) * (x - x_bar) for y, x in list(zip(Y, X))])
        #평균과의 차이 y = wx + b 하나의 뉴런, for 다층뉴런
        a /=sum([(x - x_bar)**2 for x in X])#마이너스이니까 제곱을 해줌
        b = y_bar - a * x_bar
        # y = wx + b
        print(f'a : {a}, b:{b}')

        line_x = np.arange(min(X), max(X), 0.01)
        line_y = a * line_x + b#가중치[W]가 계속 바뀌는 것 : 경사하강법, line_x는 바뀌지 않음, 이것이 뉴런
        return {f'line_x : {line_x}, line_y :{line_y}'}

    def population_with_regression(self):
        # 4.3 최소제곱법으로 회귀선 구하기
        X = self.population_inc
        Y = self.population_old
        # X, Y의 평균
        x_bar = sum(X) / len(X)
        y_bar = sum(Y) / len(Y)
        # 최소제곱법으로 a, b를 구합니다.
        a = sum([(y - y_bar) * (x - x_bar) for y, x in list(zip(Y, X))])
        a /= sum([(x - x_bar) ** 2 for x in X])
        b = y_bar - a * x_bar
        print('a:', a, 'b:', b)
        # 그래프를 그리기 위해 회귀선의 x, y 데이터를 구합니다.
        line_x = np.arange(min(X), max(X), 0.01)
        line_y = a * line_x + b
        return {'line_x':line_x, 'line_y': line_y}

    def population_with_regression_using_tf(self):
        X = self.population_inc
        Y = self.population_old
        a = tf.Variable(random.random())
        b = tf.Variable(random.random())
        # 잔차의 제곱의 평균을 반환하는 함수
        def compute_loss():
            y_pred = a * X + b
            loss = tf.reduce_mean((Y - y_pred) **2)
            return loss
        optimizer = tf.keras.optimizers.Adam(lr=0.07)
        for i in range(1000):
            optimizer.minimize(compute_loss, var_list=[a,b])
            if i % 100 ==99:
                print(i, 'a: ', a.numpy(), 'b: ', b.numpy(), 'loss: ', compute_loss().numpy())
        line_x = np.arange(min(X), max(X), 0.01)
        line_y = a * line_x + b
        return {'line_x':line_x, 'line_y': line_y}
    def normalization(self):
        pass

    def new_model(self):
        X = instance.population_inc
        Y = instance.population_old
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=6, activation='tanh', input_shape=(1,)),
            tf.keras.layers.Dense(units=1)
        ])
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')
        # mse : mean squared error
        model.fit(X, Y, epochs=10)
        model.predict(X)
        return model

    def predict(self, model):
    #deep learing 회귀선 ==MLP
        X = self.population_inc
        line_x = np.arange(min(X), max(X), 0.01)
        line_y = model.predict(line_x)
        return {'line_x' : line_x, 'line_y' : line_y}

class View:
    @staticmethod
    def show_population(instance, dic):
        X = instance.population_inc
        Y = instance.population_old
        line_x = dic['line_x']
        line_y = dic['line_y']
        # 붉은색 실선으로 회귀선을 그립니다.
        plt.plot(line_x, line_y, 'r-')
        plt.plot(X, Y, 'bo')
        plt.xlabel('Population Growth Rate (%)')
        plt.ylabel('Elderly Population Rate (%)')
        plt.show()

if __name__ == '__main__':##ANN : 싱글뉴런  DNN : 멀티 뉴런  CNN : 멀티 DNN[멀티텐서]
    instance = Population()
    view = View()
    instance.initialize()
    instance.population_without_outlier()
    #dic = instance.population_with_regression() # 내가 하는 것
    #dic = instance.population_with_regression_using_tf() #텐서플로를 이용하는 방법
    dic = instance.predict(instance.new_model()) #딥러닝을 기반으로 하는 방법
    view.show_population(instance, dic)
