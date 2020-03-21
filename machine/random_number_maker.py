import tensorflow as tf
import math
class RandomNumberMaker:
    def fetch_random_number(self):
        rand = tf.random.uniform([1], 0, 1) # 균일 분포
        print(rand)
        rand = tf.random.uniform([4], 0, 1)  # 여러개, 균일 분포
        print(rand)
        rand = tf.random.normal([4], 0, 1)  # 여러개, 정규 분포
        print(rand)
        return rand
    @staticmethod
    def sigmoid(x):
        return 1 / ( 1 + math.exp(-x))

    def create_neuron(self):
        x = 1
        y = 0
        w = tf.random.normal([1], 0, 1)
        output = self.sigmoid(x * w)
        print(output)
        # 경사하강법을 이용한 뉴런 학습
        for i in range(1000):
            output = self.sigmoid(x * w)
            error = y - output
            w = w + y * 0.1 * error

            if i % 100 == 99:
                print(i, error, output)

        return output


        